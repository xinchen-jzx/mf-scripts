#!/usr/bin/env python3
"""
性能实验脚本
用于测试不同block_size和batch_size对smem_bm_copy_batch性能的影响
"""

import argparse
import shlex
import signal
import subprocess
import re
import json
import os
import time
import matplotlib.pyplot as plt
from collections import defaultdict
from datetime import datetime

# 实验配置
BENCHMARK_CMD = "./bm_perf_benchmark"  # 可执行文件路径
IP_PORT = "tcp://127.0.0.1:8570"  # configStore服务器地址
WORLD_SIZE = 2  # 两卡
LOCAL_RANK_SIZE = 2  # 单节点两卡
COPY_COUNT = 100  # 拷贝次数
WARMUP_COUNT = 10  # 预热次数
GVA_SIZE = 16 * 1024 * 1024 * 1024  # 16GB，与 band_width_calculator.cpp 保持一致

# 测试模式
TEST_TYPES = ["rd2d", "d2rd", "rh2d", "d2rh"]

# 实验参数范围
# block_size: 从1024开始，每次翻2倍，直到16MB
BLOCK_SIZES = [1] + [1024 * (2 ** i) for i in range(15)]
# batch_size: 从1开始，每次翻4倍，直到4096 (4^6 = 4096)
BATCH_SIZES = [4 ** i for i in range(11)]  # [1, 4, 16, 64, 256, 1024, 4096, 16384, 65536, 262144]

# 结果存储目录
MAX_RETRIES = 3  # 单个数据点最大重试次数

RESULTS_DIR = "perf_results"
PLOTS_DIR = "perf_plots"


def extract_timestamp_from_filename(path):
    if not path:
        return None
    basename = os.path.basename(path)
    match = re.search(r"(\d{8}_\d{6})", basename)
    if match:
        return match.group(1)
    return None


def terminate_process_group(proc):
    if proc is None:
        return
    try:
        if proc.poll() is not None:
            return
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=3)
    except ProcessLookupError:
        pass
    except Exception:
        pass


def split_csv(value):
    if not value:
        return []
    return [x.strip() for x in value.split(",") if x.strip()]


def ensure_ssh_options(options):
    result = list(options)
    needed = {
        "BatchMode": "yes",
        "ConnectTimeout": "8",
    }
    existing_keys = set()
    for opt in result:
        key = opt.split("=", 1)[0].strip()
        if key:
            existing_keys.add(key)
    for key, value in needed.items():
        if key not in existing_keys:
            result.append(f"{key}={value}")
    return result


def parse_env_assignments(items):
    env_dict = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"invalid --ssh_remote_env item: {item}, expected KEY=VALUE")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"invalid --ssh_remote_env key in: {item}")
        env_dict[key] = value
    return env_dict


def short_text(text, limit=None):
    if not text:
        return ""
    text = text.strip()
    if limit is None:
        return text
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...<truncated>..."


def build_benchmark_cmd(args, copy_type, block_size, batch_size, use_mte, rank_start, device_start):
    cmd = [
        args.benchmark,
        "-bw",
        "-ot", "sdma",
        "-t", copy_type,
        "-s", str(block_size),
        "-bs", str(batch_size),
        "-cc", str(args.copy_count),
        "-ws", str(args.world_size),
        "-lrs", str(args.local_rank_size),
        "-rs", str(rank_start),
        "-d", str(device_start),
        "-ip", args.ip,
        "-solo",
        "-warmup", str(args.warmup_count)
    ]
    if use_mte:
        cmd.append("-mte")
    return cmd


def build_ssh_command(args, remote_cmd):
    ssh_cmd = ["ssh", "-n"]
    if args.ssh_port > 0:
        ssh_cmd.extend(["-p", str(args.ssh_port)])
    if args.ssh_key:
        ssh_cmd.extend(["-i", args.ssh_key])
    for opt in args.ssh_options:
        ssh_cmd.extend(["-o", opt])
    target = f"{args.ssh_user}@{args.ssh_host}" if args.ssh_user else args.ssh_host

    remote_parts = []
    if args.ssh_remote_workdir:
        remote_parts.append(f"cd {shlex.quote(args.ssh_remote_workdir)}")
    remote_env = parse_env_assignments(args.ssh_remote_env)
    env_prefix = " ".join(f"{k}={shlex.quote(v)}" for k, v in remote_env.items())
    remote_cmd_str = shlex.join(remote_cmd)
    if env_prefix:
        remote_cmd_str = f"{env_prefix} {remote_cmd_str}"
    remote_parts.append(remote_cmd_str)
    remote_script = " && ".join(remote_parts)
    ssh_cmd.extend([target, f"bash -lc {shlex.quote(remote_script)}"])
    return ssh_cmd


def validate_ssh_args(args):
    args.ssh_enabled = bool(args.ssh_host)
    if not args.ssh_enabled:
        return
    expected_world_size = args.local_rank_size + args.ssh_remote_local_rank_size
    if args.world_size != expected_world_size:
        raise ValueError(
            f"SSH模式下 world_size({args.world_size}) 必须等于 "
            f"local_rank_size({args.local_rank_size}) + "
            f"ssh_remote_local_rank_size({args.ssh_remote_local_rank_size}) = {expected_world_size}"
        )


def run_benchmark(args, copy_type, block_size, batch_size, use_mte=False):
    """运行一次benchmark测试（支持可选SSH远端协同）"""
    local_cmd = build_benchmark_cmd(
        args,
        copy_type,
        block_size,
        batch_size,
        use_mte,
        args.rank_start,
        args.device_start,
    )
    local_cmd_text = shlex.join(local_cmd)
    remote_cmd_text = ""
    remote_proc = None
    try:
        if args.ssh_enabled:
            remote_cmd = build_benchmark_cmd(
                args,
                copy_type,
                block_size,
                batch_size,
                use_mte,
                args.ssh_remote_rank_start,
                args.ssh_remote_device_start,
            )
            remote_cmd[remote_cmd.index("-lrs") + 1] = str(args.ssh_remote_local_rank_size)
            ssh_cmd = build_ssh_command(args, remote_cmd)
            remote_cmd_text = shlex.join(ssh_cmd)
            remote_proc = subprocess.Popen(
                ssh_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
            # 远端若瞬时失败（认证、路径、命令错误），这里立即暴露，避免本地卡住等待超时
            startup_deadline = time.time() + args.ssh_startup_check_seconds
            while time.time() < startup_deadline:
                if remote_proc.poll() is not None:
                    break
                time.sleep(0.1)
            if remote_proc.poll() is not None:
                remote_stdout, remote_stderr = remote_proc.communicate(timeout=1)
                print(f"Remote ssh process exits early: returncode={remote_proc.returncode}")
                print(f"  remote cmd: {remote_cmd_text}")
                if remote_stderr:
                    print("  remote stderr:")
                    print(short_text(remote_stderr))
                if remote_stdout:
                    print("  remote stdout:")
                    print(short_text(remote_stdout))
                return None

        local_proc = subprocess.Popen(
            local_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        result = None
        start_ts = time.time()
        while True:
            if remote_proc is not None and remote_proc.poll() not in (None, 0):
                terminate_process_group(local_proc)
                local_stdout = ""
                local_stderr = ""
                try:
                    local_stdout, local_stderr = local_proc.communicate(timeout=1)
                except subprocess.TimeoutExpired:
                    terminate_process_group(local_proc)
                remote_stdout, remote_stderr = remote_proc.communicate(timeout=1)
                print(f"Remote ssh benchmark failed early: returncode={remote_proc.returncode}")
                print(f"  remote cmd: {remote_cmd_text}")
                if remote_stderr:
                    print("  remote stderr:")
                    print(short_text(remote_stderr))
                if remote_stdout:
                    print("  remote stdout:")
                    print(short_text(remote_stdout))
                if local_stderr:
                    print("  local stderr:")
                    print(short_text(local_stderr))
                if local_stdout:
                    print("  local stdout:")
                    print(short_text(local_stdout))
                return None
            if local_proc.poll() is not None:
                break
            if time.time() - start_ts > args.timeout:
                terminate_process_group(local_proc)
                raise subprocess.TimeoutExpired(local_cmd, args.timeout)
            time.sleep(0.2)

        local_stdout, local_stderr = local_proc.communicate(timeout=1)
        result = subprocess.CompletedProcess(
            args=local_cmd,
            returncode=local_proc.returncode,
            stdout=local_stdout,
            stderr=local_stderr,
        )
        remote_stdout = ""
        remote_stderr = ""
        if remote_proc is not None:
            try:
                remote_stdout, remote_stderr = remote_proc.communicate(timeout=args.timeout + 30)
            except subprocess.TimeoutExpired:
                terminate_process_group(remote_proc)
                remote_stdout = remote_proc.stdout.read() if remote_proc.stdout else ""
                remote_stderr = remote_proc.stderr.read() if remote_proc.stderr else ""
        if result.returncode != 0:
            print(f"Error running local benchmark: returncode={result.returncode}")
            print(f"  local cmd: {local_cmd_text}")
            if result.stderr:
                print("  local stderr:")
                print(short_text(result.stderr))
            if result.stdout:
                print("  local stdout:")
                print(short_text(result.stdout))
            if remote_proc is not None and remote_proc.returncode not in (0, None):
                print(f"  remote ssh failed: returncode={remote_proc.returncode}")
                print(f"  remote cmd: {remote_cmd_text}")
                if remote_stderr:
                    print("  remote stderr:")
                    print(short_text(remote_stderr))
                if remote_stdout:
                    print("  remote stdout:")
                    print(short_text(remote_stdout))
            return None
        if remote_proc is not None and remote_proc.returncode not in (0, None):
            print(f"Remote ssh benchmark failed: returncode={remote_proc.returncode}")
            print(f"  remote cmd: {remote_cmd_text}")
            if remote_stderr:
                print("  remote stderr:")
                print(short_text(remote_stderr))
            if remote_stdout:
                print("  remote stdout:")
                print(short_text(remote_stdout))
            return None
        return result.stdout
    except subprocess.TimeoutExpired:
        mte_str = " (MTE)" if use_mte else ""
        print(f"Timeout running benchmark for {copy_type}, block_size={block_size}, batch_size={batch_size}{mte_str}")
        print(f"  local cmd: {local_cmd_text}")
        if remote_cmd_text:
            print(f"  remote cmd: {remote_cmd_text}")
        return None
    except (OSError, ValueError) as e:
        print(f"Exception running benchmark: {e}")
        print(f"  local cmd: {local_cmd_text}")
        if remote_cmd_text:
            print(f"  remote cmd: {remote_cmd_text}")
        return None
    finally:
        terminate_process_group(remote_proc)


def parse_output(output, copy_type, block_size, batch_size, use_mte=False):
    """解析benchmark输出，提取单种类型的性能数据"""
    if not output:
        return None
    
    # 查找对应类型的行
    lines = output.split('\n')
    for line in lines:
        # 匹配输出格式: Type NPU Rank TotalSize(KB) TotalTime(us) AvgSize(B) AvgTime(us) BW(GB/s) WrongBytes(B)
        # 例如: RD2D       0         0 2097152       1234567       2097152     1234.56     1.234    0
        pattern = rf'^\s*{copy_type.upper()}\s+\d+\s+\d+\s+(\d+)\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+\d+'
        match = re.match(pattern, line)
        if match:
            total_size_kb = int(match.group(1))
            total_time_us = int(match.group(2))
            avg_size_b = int(match.group(3))
            avg_time_us = float(match.group(4))
            bandwidth_gbps = float(match.group(5))
            
            # 计算IOPS: batch_size / 单次copy执行时间(s)
            iops = batch_size * 1000000.0 / avg_time_us if avg_time_us > 0 else 0
            
            return {
                'copy_type': copy_type.upper(),
                'block_size': block_size,
                'batch_size': batch_size,
                'use_mte': use_mte,
                'total_size_kb': total_size_kb,
                'total_time_us': total_time_us,
                'avg_size_b': avg_size_b,
                'avg_time_us': avg_time_us,
                'bandwidth_gbps': bandwidth_gbps,
                'iops': iops,
                'latency_us': avg_time_us
            }
    
    return None


def parse_output_all(output, block_size, batch_size, use_mte=False):
    """解析 -t all 的一次运行输出，提取六种模式（RD2D,D2RD,H2D,D2H,RH2D,D2RH）的性能数据"""
    if not output:
        return None
    results_by_type = {}
    for copy_type in TEST_TYPES:
        r = parse_output(output, copy_type, block_size, batch_size, use_mte)
        if r:
            results_by_type[copy_type] = r
    # 必须六种都解析到才认为成功
    if len(results_by_type) == len(TEST_TYPES):
        return results_by_type
    return None


def run_all_experiments(args):
    """运行所有实验"""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # 生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    all_results = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    
    # 每种配置只跑一次 -t all，从输出中解析六种模式（RD2D,D2RD,H2D,D2H,RH2D,D2RH）
    total_experiments = len(BLOCK_SIZES) * len(BATCH_SIZES) * 2  # 每个配置跑 2 次：不开MTE + 开MTE
    current = 0
    
    print(f"开始运行实验，共 {total_experiments} 个配置（每配置 -t all 一次，解析六种模式）...")
    
    for block_size in BLOCK_SIZES:
        for batch_size in BATCH_SIZES:
            for use_mte in [False, True]:
                current += 1
                mte_str = " (MTE)" if use_mte else ""
                if block_size * batch_size > GVA_SIZE:
                    print(f"  [{current}/{total_experiments}] block_size={block_size}, batch_size={batch_size}{mte_str}... 跳过 (block_size*batch_size > GVA_SIZE)")
                    continue
                print(f"  [{current}/{total_experiments}] block_size={block_size}, batch_size={batch_size}{mte_str}...", end="", flush=True)
                
                success = False
                for attempt in range(1, MAX_RETRIES + 1):
                    local_cmd = build_benchmark_cmd(
                        args,
                        "all",
                        block_size,
                        batch_size,
                        use_mte,
                        args.rank_start,
                        args.device_start,
                    )
                    local_cmd_text = shlex.join(local_cmd)
                    remote_cmd_text = ""
                    if args.ssh_enabled:
                        remote_cmd = build_benchmark_cmd(
                            args,
                            "all",
                            block_size,
                            batch_size,
                            use_mte,
                            args.ssh_remote_rank_start,
                            args.ssh_remote_device_start,
                        )
                        remote_cmd[remote_cmd.index("-lrs") + 1] = str(args.ssh_remote_local_rank_size)
                        remote_cmd_text = shlex.join(build_ssh_command(args, remote_cmd))
                    output = run_benchmark(args, "all", block_size, batch_size, use_mte)
                    if output:
                        results_by_type = parse_output_all(output, block_size, batch_size, use_mte)
                        if results_by_type:
                            for copy_type, result in results_by_type.items():
                                all_results[copy_type][block_size][batch_size][use_mte] = result
                            if attempt == 1:
                                print(" ✓")
                            else:
                                print(f" ✓ (重试{attempt - 1}次)")
                            success = True
                            break
                        print("  解析输出失败，原始输出:")
                        print(f"  local cmd: {local_cmd_text}")
                        if remote_cmd_text:
                            print(f"  remote cmd: {remote_cmd_text}")
                        print(output.strip())
                    if attempt < MAX_RETRIES:
                        print(f" ✗ (第{attempt}次失败，重试中)", end="", flush=True)
                
                if not success:
                    print(f" ✗ (重试{MAX_RETRIES}次均失败，跳过)")
    
    # 保存结果到JSON文件（带时间戳）
    results_file = os.path.join(RESULTS_DIR, f"all_results_{timestamp}.json")
    # 转换defaultdict为普通dict以便JSON序列化
    results_dict = {}
    for copy_type, block_data in all_results.items():
        results_dict[copy_type] = {}
        for block_size, batch_data in block_data.items():
            results_dict[copy_type][str(block_size)] = {}
            for batch_size, mte_data in batch_data.items():
                results_dict[copy_type][str(block_size)][str(batch_size)] = {}
                for use_mte, result in mte_data.items():
                    results_dict[copy_type][str(block_size)][str(batch_size)][str(use_mte)] = result
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"\n实验结果已保存到: {results_file}")
    return all_results, timestamp


def load_results(results_file=None):
    """从文件加载实验结果"""
    if results_file is None:
        # 查找最新的结果文件
        if not os.path.exists(RESULTS_DIR):
            return None, None
        result_files = [f for f in os.listdir(RESULTS_DIR) if f.startswith("all_results_") and f.endswith(".json")]
        if not result_files:
            return None, None
        # 按文件名排序，取最新的
        result_files.sort(reverse=True)
        results_file = os.path.join(RESULTS_DIR, result_files[0])
        # 从文件名提取时间戳
        timestamp = extract_timestamp_from_filename(result_files[0])
    else:
        # 从文件名提取时间戳
        timestamp = extract_timestamp_from_filename(results_file)
    
    if not os.path.exists(results_file):
        return None, None
    
    with open(results_file, 'r', encoding='utf-8') as f:
        results_dict = json.load(f)
    
    # 转换回defaultdict结构
    all_results = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for copy_type, block_data in results_dict.items():
        for block_size_str, batch_data in block_data.items():
            block_size = int(block_size_str)
            for batch_size_str, mte_data in batch_data.items():
                batch_size = int(batch_size_str)
                if isinstance(mte_data, dict) and any(k in ['True', 'False'] for k in mte_data.keys()):
                    # 新格式：包含MTE数据
                    for use_mte_str, result in mte_data.items():
                        use_mte = use_mte_str == 'True'
                        all_results[copy_type][block_size][batch_size][use_mte] = result
                else:
                    # 旧格式：兼容处理，默认不开MTE
                    all_results[copy_type][block_size][batch_size][False] = mte_data
    
    return all_results, timestamp


def get_copy_type_description(copy_type):
    """根据copy_type生成描述性标题"""
    descriptions = {
        'rd2d': 'Read from HBM Pool',
        'd2rd': 'Write to HBM Pool',
        'h2d': 'Read from same rank\'s DRAM',
        'd2h': 'Write to same rank\'s DRAM',
        'rh2d': 'Read from DRAM Pool',
        'd2rh': 'Write to DRAM Pool'
    }
    return descriptions.get(copy_type.lower(), copy_type.upper())


def plot_results(all_results, timestamp=None):
    """绘制所有图表"""
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 创建带时间戳的plots目录
    plots_timestamp_dir = os.path.join(PLOTS_DIR, timestamp)
    os.makedirs(plots_timestamp_dir, exist_ok=True)
    
    metrics = {
        'iops': {'name': 'IOPS', 'ylabel': 'IOPS (ops/s)'},
        'latency_us': {'name': 'Latency', 'ylabel': 'Latency (μs)'},
        'bandwidth_gbps': {'name': 'Bandwidth', 'ylabel': 'Bandwidth (GB/s)'}
    }
    
    # 为每个batch_size分配颜色
    try:
        # 新版本matplotlib
        cmap = plt.colormaps['tab10']
    except (AttributeError, KeyError):
        # 旧版本matplotlib
        import matplotlib.cm as cm
        cmap = cm.get_cmap('tab10')
    colors = [cmap(i) for i in range(len(BATCH_SIZES))]
    batch_color_map = {bs: colors[i] for i, bs in enumerate(BATCH_SIZES)}
    
    # 为每个模式和指标生成图表
    for copy_type in TEST_TYPES:
        if copy_type not in all_results or not all_results[copy_type]:
            print(f"警告: {copy_type.upper()} 没有数据，跳过")
            continue
        
        for metric_key, metric_info in metrics.items():
            # 创建左右两个子图：左侧显示绝对性能，右侧显示加速比
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
            
            # 先收集所有值以确定y轴范围
            all_values = []
            
            # 存储每个batch_size和block_size的性能数据，用于计算比例
            performance_data = {}  # {batch_size: {block_size: {False: value, True: value}}}
            
            # ========== 左侧子图：绝对性能数据 ==========
            # 为每个batch_size和MTE状态绘制线
            for batch_size in BATCH_SIZES:
                color = batch_color_map[batch_size]
                performance_data[batch_size] = {}
                
                for use_mte in [False, True]:
                    block_sizes = []
                    values = []
                    x_positions = []
                    
                    for idx, block_size in enumerate(BLOCK_SIZES):
                        if (block_size in all_results[copy_type] and 
                            batch_size in all_results[copy_type][block_size] and
                            use_mte in all_results[copy_type][block_size][batch_size]):
                            result = all_results[copy_type][block_size][batch_size][use_mte]
                            value = result[metric_key]
                            
                            # 跳过block_size=1且bandwidth=0的数据点
                            if block_size == 1 and metric_key == 'bandwidth_gbps' and value == 0:
                                continue
                            
                            # 使用block_size在BLOCK_SIZES中的实际索引作为x轴位置
                            x_positions.append(idx)
                            block_sizes.append(block_size)
                            values.append(value)
                            all_values.append(value)
                            
                            # 存储数据用于计算比例
                            if block_size not in performance_data[batch_size]:
                                performance_data[batch_size][block_size] = {}
                            performance_data[batch_size][block_size][use_mte] = value
                    
                    if block_sizes:
                        linestyle = '-' if use_mte else '--'
                        if use_mte:
                            label = f'batch_size={batch_size} (MTE)'
                        else:
                            label = f'batch_size={batch_size} (SDMA)'
                        # 使用更粗的虚线以便区分
                        plot_kwargs = {
                            'marker': 'o',
                            'label': label,
                            'linewidth': 2.5,
                            'color': color,
                            'linestyle': linestyle,
                            'alpha': 0.8
                        }
                        if not use_mte:
                            plot_kwargs['dashes'] = (5, 5)
                        ax1.plot(x_positions, values, **plot_kwargs)
                        
                        # 在每个数据点上标注绝对数值
                        for x_pos, val in zip(x_positions, values):
                            if metric_key == 'bandwidth_gbps':
                                label_text = f'{val:.2f}'
                            elif metric_key == 'latency_us':
                                label_text = f'{val:.1f}'
                            else:
                                label_text = f'{val:.0f}'
                            ax1.annotate(label_text,
                                        xy=(x_pos, val),
                                        xytext=(0, 4), textcoords='offset points',
                                        fontsize=5, color=color, alpha=0.75,
                                        ha='center', va='bottom')
            
            # 设置左侧子图的x轴标签
            ax1.set_xlabel('Block Size', fontsize=12)
            ax1.set_ylabel(metric_info['ylabel'], fontsize=12)
            
            # 生成标题，包含描述性说明
            description = get_copy_type_description(copy_type)
            ax1.set_title(f'{description} ({copy_type.upper()}) - {metric_info["name"]} vs Block Size', 
                        fontsize=14, fontweight='bold')
            
            # 设置x轴刻度为均匀分布，标签显示实际的block_size值
            if BLOCK_SIZES:
                x_ticks = list(range(len(BLOCK_SIZES)))
                x_labels = []
                for bs in BLOCK_SIZES:
                    if bs < 1024:
                        x_labels.append(f'{bs}B')
                    elif bs < 1024 * 1024:
                        x_labels.append(f'{bs//1024}KB')
                    else:
                        x_labels.append(f'{bs//(1024*1024)}MB')
                ax1.set_xticks(x_ticks)
                ax1.set_xticklabels(x_labels, rotation=45, ha='right')
                ax2.set_xticks(x_ticks)
                ax2.set_xticklabels(x_labels, rotation=45, ha='right')
            
            from matplotlib.ticker import FuncFormatter, LogLocator

            # 设置y轴为4倍对数刻度，但标签显示实际值
            ax1.set_yscale('log', base=4)

            # 主刻度：只放在 4^k 上
            ax1.yaxis.set_major_locator(LogLocator(base=4))

            # 关键：把 tick label 强制格式化成普通数字
            def plain_number(y, _pos):
                # 按你的量级偏好改这里：下面是"尽量普通数字"的策略
                if y == 0:
                    return "0"
                if y >= 1000:
                    return f"{y:.0f}"          # 大于等于 1000 显示整数
                if y >= 1:
                    return f"{y:.2f}".rstrip('0').rstrip('.')  # 1~1000 显示到 2 位小数
                if y >= 0.001:
                    return f"{y:.3f}".rstrip('0').rstrip('.')  # 小数显示 3 位
                return f"{y:.2e}"              # 极小值用科学计数法（可改成更多位）

            ax1.yaxis.set_major_formatter(FuncFormatter(plain_number))
            
            ax1.grid(True, alpha=0.3, which='major')
            ax1.grid(True, alpha=0.15, which='minor', linestyle=':')
            
            # 左侧子图图例
            ax1.legend(loc='best', fontsize=8, framealpha=0.9, edgecolor='black', fancybox=True, ncol=2)
            
            # ========== 右侧子图：加速比 ==========
            # 计算并绘制性能提升比例
            all_ratio_values = []  # 收集所有比例值用于设置y轴范围
            
            for batch_size in BATCH_SIZES:
                if batch_size not in performance_data:
                    continue
                
                ratio_x_positions = []
                ratio_values = []
                
                for idx, block_size in enumerate(BLOCK_SIZES):
                    if block_size not in performance_data[batch_size]:
                        continue
                    data = performance_data[batch_size][block_size]
                    if False in data and True in data:
                        non_mte_value = data[False]
                        mte_value = data[True]
                        
                        # 跳过block_size=1且bandwidth=0的数据点
                        if block_size == 1 and metric_key == 'bandwidth_gbps':
                            if non_mte_value == 0 or mte_value == 0:
                                continue
                        
                        # 计算性能提升比例
                        if metric_key == 'latency_us':
                            # 对于延迟，ratio = mte / non_mte (ratio < 1表示延迟降低，MTE更好)
                            ratio = mte_value / non_mte_value if non_mte_value > 0 else 1.0
                        else:
                            # 对于IOPS和Bandwidth，ratio = mte / non_mte (ratio > 1表示MTE更好)
                            ratio = mte_value / non_mte_value if non_mte_value > 0 else 1.0
                        
                        ratio_x_positions.append(idx)
                        ratio_values.append(ratio)
                        all_ratio_values.append(ratio)
                
                if ratio_x_positions:
                    # 使用不同的标记和颜色显示性能提升比例
                    color = batch_color_map[batch_size]
                    ax2.plot(ratio_x_positions, ratio_values, marker='s', markersize=8,
                            linestyle='-', linewidth=2, color=color, alpha=0.8,
                            label=f'batch_size={batch_size}')
                    
                    # 在数据点上标注数值
                    for x_pos, ratio_val in zip(ratio_x_positions, ratio_values):
                        ax2.annotate(f'{ratio_val:.2f}x', 
                                    xy=(x_pos, ratio_val),
                                    xytext=(0, 5), textcoords='offset points',
                                    fontsize=7, color=color, alpha=0.8,
                                    ha='center', va='bottom')
            
            # 设置右侧子图的标签和标题
            ax2.set_xlabel('Block Size', fontsize=12)
            ax2.set_ylabel('MTE Speedup Ratio', fontsize=12)
            ax2.set_title(f'{description} ({copy_type.upper()}) - MTE Speedup Ratio vs Block Size', 
                         fontsize=14, fontweight='bold')
            
            # 添加1.0参考线（表示无提升）- 加粗实线
            ax2.axhline(y=1.0, color='black', linestyle='-', linewidth=3, alpha=0.8, label='No Speedup (1.0x)')
            
            # 自动调整y轴范围
            if all_ratio_values:
                ratio_min = min(all_ratio_values)
                ratio_max = max(all_ratio_values)
                ax2.set_ylim([max(0.3, ratio_min * 0.9), min(3.0, ratio_max * 1.1)])
            else:
                ax2.set_ylim([0.5, 2.0])
            
            ax2.grid(True, alpha=0.3, which='major')
            ax2.grid(True, alpha=0.15, which='minor', linestyle=':')
            
            # 右侧子图图例
            ax2.legend(loc='best', fontsize=8, framealpha=0.9, edgecolor='black', fancybox=True, ncol=2)
            
            fig.tight_layout()
            
            # 保存图片
            filename = f'{copy_type.upper()}_{metric_key}.png'
            filepath = os.path.join(plots_timestamp_dir, filename)
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            print(f"已生成图表: {filepath}")
    
    print(f"\n所有图表已保存到: {plots_timestamp_dir}/")


def parse_args():
    parser = argparse.ArgumentParser(description="运行 bm_perf_benchmark 自动化实验")
    parser.add_argument("--plot-only", action="store_true", help="仅绘图，不跑实验")
    parser.add_argument("--results-file", default="", help="指定结果JSON文件路径（仅绘图时可选）")
    parser.add_argument("legacy_results_file", nargs="?", default="", help=argparse.SUPPRESS)
    parser.add_argument("--benchmark", default=BENCHMARK_CMD, help="bm_perf_benchmark 可执行文件路径")
    parser.add_argument("--ip", default=IP_PORT, help="configStore地址，格式 tcp://<ip>:<port>")
    parser.add_argument("--world-size", type=int, default=WORLD_SIZE, dest="world_size")
    parser.add_argument("--local-rank-size", type=int, default=LOCAL_RANK_SIZE, dest="local_rank_size")
    parser.add_argument("--rank-start", type=int, default=0, dest="rank_start")
    parser.add_argument("--device-start", type=int, default=0, dest="device_start")
    parser.add_argument("--copy-count", type=int, default=COPY_COUNT, dest="copy_count")
    parser.add_argument("--warmup-count", type=int, default=WARMUP_COUNT, dest="warmup_count")
    parser.add_argument("--timeout", type=int, default=300, help="单次用例超时时间（秒）")
    parser.add_argument("--ssh-host", default="", help="远端主机地址，设置后启用跨机SSH协同")
    parser.add_argument("--ssh-user", default="", help="远端SSH用户名，留空则使用当前用户")
    parser.add_argument("--ssh-port", type=int, default=22, help="远端SSH端口")
    parser.add_argument("--ssh-key", default="", help="远端SSH私钥路径")
    parser.add_argument(
        "--ssh-options",
        default="StrictHostKeyChecking=no,UserKnownHostsFile=/dev/null",
        help="额外ssh -o选项，逗号分隔",
    )
    parser.add_argument(
        "--ssh-startup-check-seconds",
        type=int,
        default=2,
        help="远端ssh拉起后存活检查时长（秒）",
    )
    parser.add_argument("--ssh-remote-workdir", default="", help="远端执行目录（可选）")
    parser.add_argument(
        "--ssh-remote-env",
        action="append",
        default=[],
        help="远端环境变量，格式 KEY=VALUE，可重复传入",
    )
    parser.add_argument("--ssh-remote-local-rank-size", type=int, default=1, dest="ssh_remote_local_rank_size")
    parser.add_argument("--ssh-remote-rank-start", type=int, default=None, dest="ssh_remote_rank_start")
    parser.add_argument("--ssh-remote-device-start", type=int, default=0, dest="ssh_remote_device_start")
    args = parser.parse_args()
    args.ssh_options = ensure_ssh_options(split_csv(args.ssh_options))
    if args.ssh_remote_rank_start is None:
        args.ssh_remote_rank_start = args.local_rank_size
    return args


def main():
    """主函数"""
    args = parse_args()
    validate_ssh_args(args)
    timestamp = None
    if args.plot_only:
        # 仅绘图模式，从文件加载结果
        print("从文件加载实验结果...")
        selected_results_file = args.results_file or args.legacy_results_file
        if selected_results_file:
            # 指定了结果文件
            all_results, timestamp = load_results(selected_results_file)
        else:
            # 使用最新的结果文件
            all_results, timestamp = load_results()
        if all_results is None:
            print("错误: 找不到实验结果文件，请先运行实验")
            return
    else:
        # 运行实验
        all_results, timestamp = run_all_experiments(args)
        if not all_results:
            print("错误: 没有收集到任何实验结果")
            return
    
    # 绘制图表
    print("\n开始生成图表...")
    plot_results(all_results, timestamp)
    print("\n完成！")


if __name__ == "__main__":
    main()

