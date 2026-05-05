#!/usr/bin/env python3
"""
KVBlock mixed-size performance experiment runner for bm_perf_benchmark.
"""

import argparse
import json
import os
import re
import shlex
import signal
import socket
import subprocess
import time
from datetime import datetime



GVA_SIZE = 16 * 1024 * 1024 * 1024
DEFAULT_DIRECTIONS = ["d2rd", "rd2d", "rh2d", "d2rh"]

MODEL_CONFIGS = {
    "deepseek": {
        # per token: 61 * 1KB + 61 * 128B
        "token_blocks": [(1024, 61), (128, 61)],
    },
    "qwen": {
        # per token: 64 * 4KB
        "token_blocks": [(4096, 64)],
    },
}


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


def parse_tcp_endpoint(url):
    m = re.match(r"^tcp://([^:]+):(\d+)$", url.strip())
    if not m:
        return None, None
    return m.group(1), int(m.group(2))


def is_local_port_busy(host, port):
    if host not in ("127.0.0.1", "localhost"):
        return False
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(0.5)
    try:
        return s.connect_ex(("127.0.0.1", port)) == 0
    finally:
        s.close()


def find_free_local_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def replace_tcp_port(url, new_port):
    m = re.match(r"^tcp://([^:]+):(\d+)$", url.strip())
    if not m:
        return url
    return f"tcp://{m.group(1)}:{new_port}"


def split_csv(value):
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


def parse_int_list(value):
    return [int(x) for x in split_csv(value)]


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


def build_layout(model, kvblock_size, seq_len):
    if model not in MODEL_CONFIGS:
        raise ValueError(f"unsupported model: {model}")
    if seq_len % kvblock_size != 0:
        raise ValueError(f"seq_len ({seq_len}) must be divisible by kvblock_size ({kvblock_size})")

    token_blocks = MODEL_CONFIGS[model]["token_blocks"]
    blocks_per_token = sum(cnt for _, cnt in token_blocks)
    kvblocks = seq_len // kvblock_size
    batch_size = kvblocks * blocks_per_token

    mix_parts = []
    bytes_per_iter = 0
    for unit_size, count in token_blocks:
        scaled = unit_size * kvblock_size
        mix_parts.append(f"{scaled}:{count}")
        bytes_per_iter += scaled * count * kvblocks

    return {
        "batch_size": batch_size,
        "mix_spec": ",".join(mix_parts),
        "bytes_per_iter": bytes_per_iter,
    }


def parse_benchmark_output(output, direction):
    pattern = re.compile(
        rf"^\s*{direction.upper()}\s+\d+\s+\d+\s+(\d+)\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+(-?\d+)\s*$"
    )
    for line in output.splitlines():
        m = pattern.match(line)
        if not m:
            continue
        total_size_kb = int(m.group(1))
        total_time_us = int(m.group(2))
        avg_size_b = int(m.group(3))
        avg_time_us = float(m.group(4))
        bandwidth_gbps = float(m.group(5))
        wrong_bytes = int(m.group(6))
        return {
            "total_size_kb": total_size_kb,
            "total_time_us": total_time_us,
            "avg_size_b": avg_size_b,
            "avg_time_us": avg_time_us,
            "bandwidth_gbps": bandwidth_gbps,
            "wrong_bytes": wrong_bytes,
        }
    return None


def short_text(text, limit=800):
    if not text:
        return ""
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...<truncated>..."


def extract_timestamp_from_filename(path):
    if not path:
        return None
    basename = os.path.basename(path)
    match = re.search(r"(\d{8}_\d{6})", basename)
    if match:
        return match.group(1)
    return None


def build_case_cmd(
    args,
    model,
    kvblock_size,
    seq_len,
    direction,
    use_mte,
    rank_start,
    device_start,
    local_rank_size,
):
    layout = build_layout(model, kvblock_size, seq_len)
    # worst-case sparse buffer estimate for stride spacing
    sparse_bytes = int(layout["bytes_per_iter"] * args.stride)
    if sparse_bytes > GVA_SIZE:
        return None, layout, {
            "status": "skipped",
            "reason": "stride-adjusted buffer size exceeds GVA_SIZE",
            "model": model,
            "kvblock_size": kvblock_size,
            "seq_len": seq_len,
            "direction": direction,
            "op_type": "sdma",
            "use_mte": use_mte,
            **layout,
        }
    cmd = [
        args.benchmark,
        "-bw",
        "-ot",
        "sdma",
        "-t",
        direction,
        "-s",
        "1",
        "-mix",
        layout["mix_spec"],
        "-bs",
        str(layout["batch_size"]),
        "-st",
        str(args.stride),
        "-cc",
        str(args.copy_count),
        "-ws",
        str(args.world_size),
        "-lrs",
        str(local_rank_size),
        "-rs",
        str(rank_start),
        "-d",
        str(device_start),
        "-ip",
        args.ip,
        "-solo",
        "-warmup",
        str(args.warmup),
    ]
    if use_mte:
        cmd.append("-mte")
    return cmd, layout, None


def run_one_case(args, model, kvblock_size, seq_len, direction, use_mte):
    cmd, layout, skipped = build_case_cmd(
        args,
        model,
        kvblock_size,
        seq_len,
        direction,
        use_mte,
        args.rank_start,
        args.device_start,
        args.local_rank_size,
    )
    if skipped:
        return skipped

    local_cmd_text = shlex.join(cmd)
    remote_cmd_text = ""
    remote_proc = None
    local_proc = None
    try:
        if args.ssh_enabled:
            remote_cmd, _, _ = build_case_cmd(
                args,
                model,
                kvblock_size,
                seq_len,
                direction,
                use_mte,
                args.ssh_remote_rank_start,
                args.ssh_remote_device_start,
                args.ssh_remote_local_rank_size,
            )
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
            startup_deadline = time.time() + args.ssh_startup_check_seconds
            while time.time() < startup_deadline:
                if remote_proc.poll() is not None:
                    break
                time.sleep(0.1)
            if remote_proc.poll() is not None:
                remote_stdout, remote_stderr = remote_proc.communicate(timeout=1)
                return {
                    "status": "failed",
                    "reason": f"ssh_remote_start_failed={remote_proc.returncode}",
                    "stderr": remote_stderr,
                    "stdout": remote_stdout,
                    "cmd": local_cmd_text,
                    "ssh_cmd": remote_cmd_text,
                    "model": model,
                    "kvblock_size": kvblock_size,
                    "seq_len": seq_len,
                    "direction": direction,
                    "op_type": "sdma",
                    "use_mte": use_mte,
                    **layout,
                }

        local_proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
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
                return {
                    "status": "failed",
                    "reason": f"ssh_remote_returncode={remote_proc.returncode}",
                    "stderr": remote_stderr,
                    "stdout": remote_stdout,
                    "local_stderr": local_stderr,
                    "local_stdout": local_stdout,
                    "cmd": local_cmd_text,
                    "ssh_cmd": remote_cmd_text,
                    "model": model,
                    "kvblock_size": kvblock_size,
                    "seq_len": seq_len,
                    "direction": direction,
                    "op_type": "sdma",
                    "use_mte": use_mte,
                    **layout,
                }
            if local_proc.poll() is not None:
                break
            if time.time() - start_ts > args.timeout:
                terminate_process_group(local_proc)
                raise subprocess.TimeoutExpired(cmd, args.timeout)
            time.sleep(0.2)

        stdout, stderr = local_proc.communicate(timeout=1)
        remote_stdout = ""
        remote_stderr = ""
        if remote_proc is not None:
            try:
                remote_stdout, remote_stderr = remote_proc.communicate(timeout=args.timeout + 30)
            except subprocess.TimeoutExpired:
                terminate_process_group(remote_proc)
                remote_stdout = remote_proc.stdout.read() if remote_proc.stdout else ""
                remote_stderr = remote_proc.stderr.read() if remote_proc.stderr else ""
    except subprocess.TimeoutExpired:
        terminate_process_group(local_proc)
        terminate_process_group(remote_proc)
        return {
            "status": "failed",
            "reason": "timeout",
            "cmd": local_cmd_text,
            "ssh_cmd": remote_cmd_text,
            "model": model,
            "kvblock_size": kvblock_size,
            "seq_len": seq_len,
            "direction": direction,
            "op_type": "sdma",
            "use_mte": use_mte,
            **layout,
        }
    except KeyboardInterrupt:
        terminate_process_group(local_proc)
        terminate_process_group(remote_proc)
        raise
    except Exception as e:
        terminate_process_group(local_proc)
        terminate_process_group(remote_proc)
        return {
            "status": "failed",
            "reason": f"exception:{e}",
            "cmd": local_cmd_text,
            "ssh_cmd": remote_cmd_text,
            "model": model,
            "kvblock_size": kvblock_size,
            "seq_len": seq_len,
            "direction": direction,
            "op_type": "sdma",
            "use_mte": use_mte,
            **layout,
        }

    if local_proc.returncode != 0:
        return {
            "status": "failed",
            "reason": f"returncode={local_proc.returncode}",
            "stderr": stderr,
            "stdout": stdout,
            "cmd": local_cmd_text,
            "ssh_cmd": remote_cmd_text,
            "model": model,
            "kvblock_size": kvblock_size,
            "seq_len": seq_len,
            "direction": direction,
            "op_type": "sdma",
            "use_mte": use_mte,
            **layout,
        }
    if remote_proc is not None and remote_proc.returncode not in (0, None):
        return {
            "status": "failed",
            "reason": f"ssh_remote_returncode={remote_proc.returncode}",
            "stderr": remote_stderr,
            "stdout": remote_stdout,
            "cmd": local_cmd_text,
            "ssh_cmd": remote_cmd_text,
            "model": model,
            "kvblock_size": kvblock_size,
            "seq_len": seq_len,
            "direction": direction,
            "op_type": "sdma",
            "use_mte": use_mte,
            **layout,
        }

    parsed = parse_benchmark_output(stdout, direction)
    if not parsed:
        return {
            "status": "failed",
            "reason": "parse_output_failed",
            "stdout": stdout,
            "cmd": local_cmd_text,
            "ssh_cmd": remote_cmd_text,
            "model": model,
            "kvblock_size": kvblock_size,
            "seq_len": seq_len,
            "direction": direction,
            "op_type": "sdma",
            "use_mte": use_mte,
            **layout,
        }
    parsed["iops"] = layout["batch_size"] * 1e6 / parsed["avg_time_us"] if parsed["avg_time_us"] > 0 else 0.0

    return {
        "status": "ok",
        "model": model,
        "kvblock_size": kvblock_size,
        "seq_len": seq_len,
        "direction": direction,
        "op_type": "sdma",
        "use_mte": use_mte,
        **layout,
        **parsed,
    }


def print_case_failure(item):
    reason = item.get("reason", "unknown")
    print(f"    失败原因: {reason}")
    if item.get("cmd"):
        print(f"    本地命令: {item['cmd']}")
    if item.get("ssh_cmd"):
        print(f"    远端命令: {item['ssh_cmd']}")
    if item.get("stderr"):
        print("    stderr:")
        print(short_text(item.get("stderr")))
    if item.get("stdout"):
        print("    stdout:")
        print(short_text(item.get("stdout")))
    if item.get("local_stderr"):
        print("    local stderr:")
        print(short_text(item.get("local_stderr")))
    if item.get("local_stdout"):
        print("    local stdout:")
        print(short_text(item.get("local_stdout")))


def save_results(results, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_json = os.path.join(out_dir, f"perf_results_{ts}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    return out_json, ts


def load_results(input_json):
    with open(input_json, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_results(results, plots_dir, ts):
    import matplotlib.pyplot as plt
    from matplotlib import cm

    os.makedirs(plots_dir, exist_ok=True)
    target_dir = os.path.join(plots_dir, ts)
    os.makedirs(target_dir, exist_ok=True)

    ok_items = [x for x in results if x.get("status") == "ok"]
    if not ok_items:
        print("没有可绘图的成功结果。")
        return target_dir

    models = sorted(set(x["model"] for x in ok_items))
    directions = sorted(set(x["direction"] for x in ok_items))
    metrics = [
        ("bandwidth_gbps", "Bandwidth (GB/s)"),
        ("avg_time_us", "Latency (us)"),
        ("iops", "IOPS (ops/s)"),
    ]

    def format_metric_value(metric_key, value):
        if metric_key == "bandwidth_gbps":
            return f"{value:.2f}"
        if metric_key == "avg_time_us":
            return f"{value:.2f}"
        if metric_key == "iops":
            return f"{value:.0f}"
        return f"{value:.2f}"

    for model in models:
        model_items = [x for x in ok_items if x["model"] == model]
        kvblocks = sorted(set(x["kvblock_size"] for x in model_items))
        seq_lens = sorted(set(x["seq_len"] for x in model_items))
        # Keep a stable color per kvblock_size so SDMA/MTE are visually paired.
        cmap = cm.get_cmap("tab20", max(len(kvblocks), 1))
        kv_colors = {kv: cmap(i) for i, kv in enumerate(kvblocks)}
        for direction in directions:
            for metric_key, ylabel in metrics:
                plt.figure(figsize=(11, 7))
                for kv in kvblocks:
                    for use_mte in [False, True]:
                        xs = []
                        ys = []
                        for seq in seq_lens:
                            hit = None
                            for item in model_items:
                                if (
                                    item["direction"] == direction
                                    and item["kvblock_size"] == kv
                                    and item["seq_len"] == seq
                                    and item.get("use_mte") is use_mte
                                ):
                                    hit = item
                                    break
                            if hit:
                                xs.append(seq)
                                ys.append(hit[metric_key])
                        if xs:
                            linestyle = "-" if use_mte else "--"
                            mode = "MTE" if use_mte else "SDMA"
                            label = f"kv={kv}, {mode}"
                            line_color = kv_colors[kv]
                            plt.plot(
                                xs,
                                ys,
                                marker="o",
                                linestyle=linestyle,
                                color=line_color,
                                label=label,
                            )
                            for x, y in zip(xs, ys):
                                plt.annotate(
                                    format_metric_value(metric_key, y),
                                    xy=(x, y),
                                    xytext=(0, 6),
                                    textcoords="offset points",
                                    ha="center",
                                    va="bottom",
                                    fontsize=7,
                                    color=line_color,
                                )

                plt.title(f"{model} {direction.upper()} {metric_key}")
                plt.xlabel("seq_len")
                plt.ylabel(ylabel)
                plt.grid(alpha=0.3)
                plt.legend()
                out_file = os.path.join(target_dir, f"{model}_{direction}_{metric_key}.png")
                plt.tight_layout()
                plt.savefig(out_file, dpi=200)
                plt.close()
    return target_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Run mixed block-size kvblock experiments.")
    parser.add_argument("--benchmark", default="./bm_perf_benchmark", help="benchmark binary path")
    parser.add_argument("--ip", default="tcp://127.0.0.1:8570", help="config store ip:port")
    parser.add_argument("--world-size", "--world_size", dest="world_size", type=int, default=2)
    parser.add_argument("--local-rank-size", "--local_rank_size", dest="local_rank_size", type=int, default=2)
    parser.add_argument("--rank-start", "--rank_start", dest="rank_start", type=int, default=0)
    parser.add_argument("--device-start", "--device_start", dest="device_start", type=int, default=0)
    parser.add_argument("--copy-count", "--copy_count", dest="copy_count", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--models", default="deepseek,qwen")
    parser.add_argument("--kvblock-sizes", "--kvblock_sizes", dest="kvblock_sizes", default="8,16,32,64,128")
    parser.add_argument("--seq-lengths", "--seq_lengths", dest="seq_lengths", default="2048,4096,8192")
    parser.add_argument("--directions", default="d2rd,rd2d,rh2d,d2rh")
    parser.add_argument("--results-dir", "--results_dir", dest="results_dir", default="perf_results")
    parser.add_argument("--plots-dir", "--plots_dir", dest="plots_dir", default="perf_plots")
    parser.add_argument("--plot-only", "--plot_only", dest="plot_only", action="store_true")
    parser.add_argument("--input-json", "--input_json", dest="input_json", default="")
    parser.add_argument(
        "--auto-change-port-if-busy",
        "--auto_change_port_if_busy",
        action="store_true",
        dest="auto_change_port_if_busy",
        default=True,
        help="auto switch to a free localhost port when --ip port is busy",
    )
    parser.add_argument(
        "--no-auto-change-port-if-busy",
        "--no_auto_change_port_if_busy",
        action="store_true",
        dest="no_auto_change_port_if_busy",
        help="disable auto switch and fail when --ip port is busy",
    )
    parser.add_argument("--ssh-host", "--ssh_host", dest="ssh_host", default="", help="远端主机地址，设置后启用跨机SSH协同")
    parser.add_argument("--ssh-user", "--ssh_user", dest="ssh_user", default="", help="远端SSH用户名，留空则使用当前用户")
    parser.add_argument("--ssh-port", "--ssh_port", dest="ssh_port", type=int, default=22, help="远端SSH端口")
    parser.add_argument("--ssh-key", "--ssh_key", dest="ssh_key", default="", help="远端SSH私钥路径")
    parser.add_argument(
        "--ssh-options",
        "--ssh_options",
        dest="ssh_options",
        default="StrictHostKeyChecking=no,UserKnownHostsFile=/dev/null",
        help="额外ssh -o选项，逗号分隔",
    )
    parser.add_argument(
        "--ssh-startup-check-seconds",
        "--ssh_startup_check_seconds",
        dest="ssh_startup_check_seconds",
        type=int,
        default=2,
        help="远端ssh拉起后存活检查时长（秒）",
    )
    parser.add_argument("--ssh-remote-workdir", "--ssh_remote_workdir", dest="ssh_remote_workdir", default="", help="远端执行目录（可选）")
    parser.add_argument(
        "--ssh-remote-env",
        "--ssh_remote_env",
        dest="ssh_remote_env",
        action="append",
        default=[],
        help="远端环境变量，格式 KEY=VALUE，可重复传入",
    )
    parser.add_argument(
        "--ssh-remote-local-rank-size",
        "--ssh_remote_local_rank_size",
        dest="ssh_remote_local_rank_size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--ssh-remote-rank-start",
        "--ssh_remote_rank_start",
        dest="ssh_remote_rank_start",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--ssh-remote-device-start",
        "--ssh_remote_device_start",
        dest="ssh_remote_device_start",
        type=int,
        default=0,
    )
    args = parser.parse_args()
    args.ssh_options = ensure_ssh_options(split_csv(args.ssh_options))
    if args.ssh_remote_rank_start is None:
        args.ssh_remote_rank_start = args.local_rank_size
    return args


def main():
    args = parse_args()
    validate_ssh_args(args)
    if args.plot_only:
        if not args.input_json:
            raise ValueError("--plot-only requires --input-json")
        loaded = load_results(args.input_json)
        ts = extract_timestamp_from_filename(args.input_json) or datetime.now().strftime("%Y%m%d_%H%M%S")
        plots_path = plot_results(loaded, args.plots_dir, ts)
        print(f"图表输出目录: {plots_path}")
        return

    models = split_csv(args.models)
    kvblock_sizes = parse_int_list(args.kvblock_sizes)
    seq_lengths = parse_int_list(args.seq_lengths)
    directions = split_csv(args.directions)

    host, port = parse_tcp_endpoint(args.ip)
    if host is None:
        raise ValueError(f"invalid --ip format: {args.ip}, expected tcp://<ip>:<port>")
    auto_switch = args.auto_change_port_if_busy and (not args.no_auto_change_port_if_busy)
    if is_local_port_busy(host, port):
        if auto_switch:
            new_port = find_free_local_port()
            old_ip = args.ip
            args.ip = replace_tcp_port(args.ip, new_port)
            print(f"检测到端口占用，自动切换 configStore 端口: {old_ip} -> {args.ip}")
        else:
            raise RuntimeError(
                f"检测到端口 {port} 已被占用，通常是上次中断后残留 bm_perf_benchmark 进程导致。\n"
                "请先清理：pkill -9 -f bm_perf_benchmark\n"
                f"然后重试当前命令（--ip {args.ip}）。"
            )

    for model in models:
        if model not in MODEL_CONFIGS:
            raise ValueError(f"unsupported model: {model}, supported: {','.join(MODEL_CONFIGS.keys())}")
    for d in directions:
        if d not in DEFAULT_DIRECTIONS:
            raise ValueError(f"unsupported direction: {d}, supported: {','.join(DEFAULT_DIRECTIONS)}")

    total = len(models) * len(kvblock_sizes) * len(seq_lengths) * len(directions) * 2
    print(f"总任务数: {total}")
    results = []
    idx = 0
    try:
        for model in models:
            for kv in kvblock_sizes:
                for seq in seq_lengths:
                    if seq % kv != 0:
                        print(f"跳过 model={model} kv={kv} seq={seq}: seq_len 不能被 kvblock_size 整除")
                        continue
                    for direction in directions:
                        for use_mte in [False, True]:
                            idx += 1
                            mode = "MTE" if use_mte else "SDMA"
                            print(
                                f"[{idx}/{total}] model={model} kv={kv} seq={seq} dir={direction} mode={mode}",
                                end=" ... ",
                                flush=True,
                            )
                            item = run_one_case(args, model, kv, seq, direction, use_mte)
                            results.append(item)
                            print(item["status"])
                            if item["status"] == "failed":
                                print_case_failure(item)
    except KeyboardInterrupt:
        print(
            "\n收到 Ctrl-C，中断实验。若后续再次出现首个用例卡住，请先执行：\n"
            "pkill -9 -f bm_perf_benchmark"
        )
        raise

    out_json, ts = save_results(results, args.results_dir)
    print(f"结果文件: {out_json}")
    plots_path = plot_results(results, args.plots_dir, ts)
    print(f"图表输出目录: {plots_path}")


if __name__ == "__main__":
    main()
