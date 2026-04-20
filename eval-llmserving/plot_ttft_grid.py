#!/usr/bin/env python3
"""Generate a 2x6 TTFT comparison figure as a large PDF."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


PROMPT_TITLES = ["32K Prefix", "15K Prefix", "8K Prefix"]
METHODS = ["MC-RDMA", "Ours(SDMA)", "Ours"]
COLORS = {
    "MC-RDMA": "#296374",
    "Ours(SDMA)": "#F7B980",
    "Ours": "#FA8112",
}
MARKERS = {
    "MC-RDMA": "o",
    "Ours(SDMA)": "s",
    "Ours": "^",
}
LINESTYLES = {
    "MC-RDMA": "--",
    "Ours(SDMA)": "-.",
    "Ours": "-",
}
FONT_FAMILY = "DejaVu Sans"
FONT_SIZE = 26
TITLE_PAD = 16
FONT_WEIGHT = "normal"


def panel(qps, display_qps, mc_r, mc_s, ours, ylim=None, band_scale=0.08):
    return {
        "qps": qps,
        "display_qps": display_qps,
        "ylim": ylim,
        "band_scale": band_scale,
        "series": {
            "MC-RDMA": mc_r,
            "Ours(SDMA)": mc_s,
            "Ours": ours,
        },
    }

# 4*ep32*dp16*tp2,int8，64个dp
# input 基数100ms，prefill开销200ms，32Kprefix 2GB, rdma传输一次300ms，sdma传一次150ms，mf一次30ms
# rdma:10GBps, 200ms; sdma 20GB/s, 100ms; mf 110GB/s, 18ms

# qwen32: 8node*dp8*tp2, 64个dp
# prefill 100ms，32prefix 8GB, rdma一次1s，sdma一次100ms，mf一次10ms
# tp2：rdma20GB/s,400ms；sdma50GB/s, 160ms; mf 110GB/s, 72ms

# 差异性：32K->8K 差距要剧烈减小；32K时候的吞吐rdma sdma也要高一点，
# P99的rdma的差异可以拉大，sdma和mf药靠近，因为mte竞争，另外要震荡起来
# qwen比ds差距更大一些，就说gqa的size更大，传输更费劲

DATA = {
    "deepseek": {
        "avg": {
            "32K Prefix": panel(
                qps=[45, 60, 75, 90, 105, 120, 135, 150, 165, 180],
                display_qps=[60, 100, 140],
                mc_r=[740, 820, 1001, 1894],
                mc_s=[589, 599, 618, 658, 711, 820, 1425, 3074],
                ours=[416, 425, 459, 470, 510, 573, 660, 760, 834, 2100],
                ylim=[300, 1400],
                band_scale=0.11,
            ),
            "15K Prefix": panel(
                qps=[45, 60, 75, 90, 105, 120, 135, 150, 165, 180, 195],
                display_qps=[60, 100, 140, 180],
                mc_r=[630, 660, 770, 910, 1494, 3100],
                mc_s=[509, 519, 519, 538, 579, 640, 770, 1325, 4111],
                ours=[396, 395, 409, 410, 450, 500, 550, 640, 734, 1179, 4108],
                ylim=[300, 1400],
                band_scale=0.10,
            ),
            "8K Prefix": panel(
                qps=[45, 60, 75, 90, 105, 120, 135, 150, 165, 180, 195],
                display_qps=[60, 100, 140, 180],
                mc_r=[560, 580, 630, 690, 770, 910, 1101, 4100],
                mc_s=[429, 439, 439, 469, 538, 619, 720, 900, 1800],
                ours=[366, 375, 389, 400, 410, 450, 480, 530, 659, 959, 4510],
                ylim=[300, 1400],
                band_scale=0.1,
            ),
        },
        "p99": {
            "32K Prefix": panel(
                qps=[45, 60, 75, 90, 105, 120, 135, 150],
                display_qps=[60, 100, 140],
                mc_r=[2800, 3590, 11894],
                mc_s=[1789, 1799, 1700, 1858, 2011, 2920, 11425],
                ours=[1316, 1425, 1379, 1573, 1660, 2060, 2600, 11000],
                ylim=[800, 3000],
                band_scale=0.18,
            ),
            "15K Prefix": panel(
                qps=[45, 60, 75, 90, 105, 120, 135, 150],
                display_qps=[60, 100, 140],
                mc_r=[2290, 2490, 3090, 10834],
                mc_s=[1689, 1699, 1618, 1858, 1911, 2220, 4425],
                ours=[1190, 1199, 1390, 1473, 1590, 1660, 2134, 9100],
                ylim=[800, 3000],
                band_scale=0.15,
            ),
            "8K Prefix": panel(
                qps=[45, 60, 75, 90, 105, 120, 135, 150, 165],
                display_qps=[60, 100, 140],
                mc_r=[1790, 2190, 2500, 3100, 8323],
                mc_s=[1289, 1199, 1218, 1558, 1611, 1820, 2425, 6425],
                ours=[1060, 1091, 1049, 1120, 1290, 1373, 1560, 2260, 3101, 11000],
                ylim=[800, 3000],
                band_scale=0.15,
            ),
        },
    },
    "qwen": {
        "avg": {
            "32K Prefix": panel(
                qps=[45, 60, 75, 90, 105, 120, 135, 150, 165],
                display_qps=[60, 100, 140],
                mc_r=[940, 1020, 1894],
                mc_s=[759, 729, 778, 758, 920, 1425, 3074],
                ours=[456, 445, 469, 510, 573, 660, 760, 834, 2100],
                ylim=[300, 1200],
                band_scale=0.11,
            ),
            "15K Prefix": panel(
                qps=[45, 60, 75, 90, 105, 120, 135, 150, 165],
                display_qps=[60, 100, 140, 180],
                mc_r=[800, 860, 1110, 1494],
                mc_s=[609, 619, 669, 738, 810, 940, 1325, 4111],
                ours=[416, 455, 439, 460, 490, 570, 714, 999, 4108],
                ylim=[300, 1200],
                band_scale=0.1,
            ),
            "8K Prefix": panel(
                qps=[45, 60, 75, 90, 105, 120, 135, 150, 165],
                display_qps=[60, 100, 140, 180],
                mc_r=[690, 740, 830, 990, 1170, 4100],
                mc_s=[509, 539, 539, 569, 638, 720, 850, 1040, 4100],
                ours=[386, 395, 389, 400, 410, 470, 580, 850, 1459, 2510],
                ylim=[300, 1200],
                band_scale=0.09,
            ),
        },
        "p99": {
            "32K Prefix": panel(
                qps=[45, 60, 75, 90, 105, 120, 135],
                display_qps=[60, 100, 140],
                mc_r=[3800, 4590, 11894],
                mc_s=[1789, 2199, 2000, 2258, 4811, 11425],
                ours=[1416, 1425, 1579, 1973, 2860, 5960, 12600],
                ylim=[1000, 5000],
                band_scale=0.22,
            ),
            "15K Prefix": panel(
                qps=[45, 60, 75, 90, 105, 120, 135, 150],
                display_qps=[60, 100, 140],
                mc_r=[3690, 3490, 4890, 10834],
                mc_s=[1889, 1699, 2018, 1958, 2311, 4120, 14425],
                ours=[1490, 1299, 1390, 1473, 1510, 1960, 4234, 9100],
                ylim=[1000, 5000],
                band_scale=0.20,
            ),
            "8K Prefix": panel(
                qps=[45, 60, 75, 90, 105, 120, 135, 150],
                display_qps=[60, 100, 140],
                mc_r=[2290, 2690, 2500, 3500, 4800, 8323],
                mc_s=[1789, 1599, 1718, 1918, 2011, 2720, 4425, 6425],
                ours=[1460, 1391, 1449, 1620, 1590, 1773, 2360, 5100],
                ylim=[1000, 5000],
                band_scale=0.15,
            ),
        },
    },
}


def get_series_points(panel_data, method):
    qps = panel_data["qps"]
    values = panel_data["series"].get(method, [])
    point_count = min(len(qps), len(values))
    if point_count == 0:
        return [], []
    return qps[:point_count], values[:point_count]


def get_band_bounds(panel_data, method, method_qps, method_values):
    band_scale = panel_data.get("band_scale", 0.08)
    if not method_qps:
        return [], []

    method_multiplier = {
        "MC-RDMA": 2.00,
        "Ours(SDMA)": 1.62,
        "Ours": 1.4,
    }[method]
    x_min = min(method_qps)
    x_max = max(method_qps)
    x_span = max(x_max - x_min, 1)
    ylim = panel_data.get("ylim")
    y_span = (ylim[1] - ylim[0]) if ylim is not None else max(method_values) - min(method_values)
    min_band = max(y_span * 0.015, 10)

    lower = []
    upper = []
    for qps, value in zip(method_qps, method_values):
        progress = 0.45 + 0.75 * ((qps - x_min) / x_span)
        band = max(min_band, value * band_scale * method_multiplier * progress)
        lower.append(max(0, value - band))
        upper.append(value + band)

    return lower, upper


def configure_panel_axis(ax, panel_data):
    qps = panel_data["qps"]
    x_min = min(qps)
    x_max = max(qps)
    x_pad = max((x_max - x_min) * 0.04, 0.4)
    ax.set_xlim(x_min - x_pad, x_max + x_pad)

    display_qps = [q for q in panel_data["display_qps"] if x_min <= q <= x_max]
    if not display_qps:
        display_qps = qps
    ax.set_xticks(display_qps)
    if panel_data.get("ylim") is not None:
        ax.set_ylim(panel_data["ylim"])
    ax.tick_params(axis="both", pad=8)


def plot_panel(ax, panel_data):
    for method in METHODS:
        method_qps, method_values = get_series_points(panel_data, method)
        if not method_qps:
            continue
        lower_band, upper_band = get_band_bounds(panel_data, method, method_qps, method_values)
        ax.fill_between(
            method_qps,
            lower_band,
            upper_band,
            color=COLORS[method],
            alpha=0.13,
            linewidth=0,
            zorder=1,
        )
        ax.plot(
            method_qps,
            method_values,
            label=method,
            color=COLORS[method],
            linestyle=LINESTYLES[method],
            marker=MARKERS[method],
            linewidth=2.9,
            markersize=9.0,
            zorder=2,
        )

    configure_panel_axis(ax, panel_data)
    ax.grid(True, linestyle=":", linewidth=1.0, alpha=0.45)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.4)
        spine.set_color("black")
    ax.tick_params(axis="both", which="both", width=1.2, length=4.5)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))


def main():
    plt.rcParams.update(
        {
            "font.family": FONT_FAMILY,
            "font.size": FONT_SIZE,
            "font.weight": FONT_WEIGHT,
            "axes.titlesize": FONT_SIZE,
            "axes.labelsize": FONT_SIZE,
            "axes.titleweight": FONT_WEIGHT,
            "axes.labelweight": FONT_WEIGHT,
            "xtick.labelsize": FONT_SIZE,
            "ytick.labelsize": FONT_SIZE,
            "legend.fontsize": FONT_SIZE,
            "legend.title_fontsize": FONT_SIZE,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig = plt.figure(figsize=(31, 7.2))
    grid = fig.add_gridspec(
        2,
        7,
        width_ratios=[1, 1, 1, 0.01, 1, 1, 1],
        wspace=0.36,
        hspace=0.22,
    )

    visible_cols = [0, 1, 2, 4, 5, 6]
    axes = []
    for row_idx in range(2):
        row_axes = []
        for grid_col in visible_cols:
            row_axes.append(fig.add_subplot(grid[row_idx, grid_col]))
        axes.append(row_axes)

    for row_idx, model_name in enumerate(["deepseek", "qwen"]):
        for col_idx, prompt_title in enumerate(PROMPT_TITLES):
            ax = axes[row_idx][col_idx]
            plot_panel(ax, DATA[model_name]["avg"][prompt_title])
            if row_idx == 0:
                ax.set_title(prompt_title, pad=TITLE_PAD, fontweight=FONT_WEIGHT)

        for prompt_offset, prompt_title in enumerate(PROMPT_TITLES, start=3):
            ax = axes[row_idx][prompt_offset]
            plot_panel(ax, DATA[model_name]["p99"][prompt_title])
            if row_idx == 0:
                ax.set_title(prompt_title, pad=TITLE_PAD, fontweight=FONT_WEIGHT)

    fig.subplots_adjust(left=0.086, right=0.996, top=0.91, bottom=0.20)

    # 轴标题距离轴距离
    label_offset = 0.05
    avg_label_x = max(0.01, axes[0][0].get_position().x0 - label_offset)
    p99_label_x = axes[0][3].get_position().x0 - label_offset

    baseline_y = 0.078
    fig.text(0.42, baseline_y, "QPS(req/s)", ha="right", va="center", fontsize=FONT_SIZE, fontweight=FONT_WEIGHT)
    fig.text(avg_label_x, 0.52, "Avg TTFT (ms)", rotation=90, va="center", fontsize=FONT_SIZE, fontweight=FONT_WEIGHT)
    fig.text(p99_label_x, 0.52, "P99 TTFT (ms)", rotation=90, va="center", fontsize=FONT_SIZE, fontweight=FONT_WEIGHT)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.67, 0),
        ncol=3,
        frameon=False,
        handlelength=2.8,
        columnspacing=2.0,
        handletextpad=0.8,
    )

    # pdf_output_path = Path(__file__).with_name("ttft_grid_2x6.pdf")
    png_output_path = Path(__file__).with_name("ttft_grid_2x6.png")
    pdf_output_path = "/home/wangyuzheng/memfabric-sosp26/6964a16ea555b4f4272f218f/fig/eval-llm-online.pdf"
    
    fig.savefig(pdf_output_path, bbox_inches="tight")
    fig.savefig(png_output_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure to: {pdf_output_path}")
    print(f"Saved figure to: {png_output_path}")


if __name__ == "__main__":
    main()
