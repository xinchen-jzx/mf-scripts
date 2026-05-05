#!/usr/bin/env python3
"""
Plot manual SDMA/MTE motivation figures.

This script generates two standalone outputs:
- left figure: one image containing two bar charts
  - left panel: weight-pull bandwidth
  - right panel: KVCache-transfer bandwidth
- right figure: standalone AVG TTFT bar chart

Fill the dictionaries below with your measured values.
"""

import argparse
import os


BANDWIDTH_YLABEL = "Bandwidth (GB/s)"
AVG_TTFT_YLABEL = "Avg. TTFT"
MEASURED_HARDWARE_BW_LIMIT_GBPS = 110

# Centralized style knobs for easy tuning.
STYLE = {
    "font.size": 18,
    "axes.labelsize": 15,
    "axes.titlesize": 20,
    "xtick.labelsize": 18,
    "ytick.labelsize": 20,
    "legend.fontsize": 18,
    # Individual subplot titles, e.g. "Weight Loading".
    "subplot_title_size": 18,
    "subplot_title_pad": 4,
    # Shared labels for the left bandwidth figure.
    "shared_xlabel_size": 16,
    "shared_xlabel_y": 0.04,
    "shared_ylabel_size": 22,
    "shared_ylabel_x": 0.1,
    # Regular axis labels, e.g. the right TTFT y label.
    "axis_label_size": 16,
    "axis_label_pad": 2,
}

MODELS = ["deepseek", "qwen"]
MODEL_DISPLAY_NAMES = {
    "deepseek": "DS",
    "qwen": "Qwen",
}
TRANSFER_TYPES = [
    ("weight_pull", "Weight Loading"),
    ("kvcache_transfer", "KVCache Transfer"),
]

# Replace these manual values with your measured numbers.
BANDWIDTH_BREAKDOWN_DATA = {
    "weight_pull": {
        "deepseek": {
            "sdma": 109.0,
            "mte": 106.0,
        },
        "qwen": {
            "sdma": 110.0,
            "mte": 108.0,
        },
    },
    "kvcache_transfer": {
        "deepseek": {
            "sdma": 21.0,
            "mte": 87.0,
        },
        "qwen": {
            "sdma": 45.0,
            "mte": 95.0,
        },
    },
}

AVG_TTFT_DATA = {
    "deepseek": {
        "sdma": 833.0,
        "mte": 1093.0,
    },
    "qwen": {
        "sdma": 651.0,
        "mte": 899.0,
    },
}

LEFT_MODEL_STYLES = {
    "deepseek": {"color": "#406AAF"},
    "qwen": {"color": "#F08D39"},
}

RIGHT_MODEL_STYLES = {
    "deepseek": {"color": "#427AB5"},
    "qwen": {"color": "#FFA02E"},
}


def validate_bar_pair_data(data, data_name):
    for category, category_data in data.items():
        for model in MODELS:
            if model not in category_data:
                raise ValueError(f"Missing {data_name} for {category}/{model}")
            for engine in ("sdma", "mte"):
                if engine not in category_data[model]:
                    raise ValueError(f"Missing {data_name} for {category}/{model}/{engine}")


def validate_avg_ttft_data(ttft_data):
    for model in MODELS:
        if model not in ttft_data:
            raise ValueError(f"Missing AVG TTFT data for model: {model}")
        for engine in ("sdma", "mte"):
            if engine not in ttft_data[model]:
                raise ValueError(f"Missing AVG TTFT data for {model}/{engine}")


def setup_style(plt):
    plt.rcParams.update(
        {
            "font.size": STYLE["font.size"],
            "axes.labelsize": STYLE["axes.labelsize"],
            "axes.titlesize": STYLE["axes.titlesize"],
            "xtick.labelsize": STYLE["xtick.labelsize"],
            "ytick.labelsize": STYLE["ytick.labelsize"],
            "legend.fontsize": STYLE["legend.fontsize"],
        }
    )


def save_figure(fig, output_stem, target_dir):
    pdf_path = os.path.join(target_dir, f"{output_stem}.pdf")
    png_path = os.path.join(target_dir, f"{output_stem}.png")
    fig.tight_layout()
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")


def plot_grouped_bars(
    ax,
    values_by_model,
    ylabel,
    title,
    model_styles,
    engine_styles,
    ylabel_size=None,
    ylabel_pad=None,
):
    x_positions = list(range(len(MODELS)))
    bar_width = 0.32
    plotted_values = []

    for idx, model in enumerate(MODELS):
        model_style = model_styles.get(model, {"color": "black"})
        for engine, offset in (("sdma", -bar_width / 2), ("mte", bar_width / 2)):
            value = values_by_model[model][engine]
            plotted_values.append(value)
            engine_style = engine_styles[engine]
            ax.bar(
                idx + offset,
                value,
                width=bar_width,
                color=model_style["color"],
                edgecolor="black",
                linewidth=1.0,
                hatch=engine_style["hatch"],
                alpha=engine_style["alpha"],
            )

    ax.set_xticks(x_positions)
    ax.set_xticklabels([MODEL_DISPLAY_NAMES[model] for model in MODELS])
    if ylabel:
        effective_ylabel_size = ylabel_size if ylabel_size is not None else STYLE["axis_label_size"]
        effective_ylabel_pad = ylabel_pad if ylabel_pad is not None else STYLE["axis_label_pad"]
        ax.set_ylabel(
            ylabel,
            fontsize=effective_ylabel_size,
            labelpad=effective_ylabel_pad,
        )
    if title:
        ax.set_title(
            title,
            fontsize=STYLE["subplot_title_size"],
            pad=STYLE["subplot_title_pad"],
        )
    ax.grid(False)

    max_value = max(plotted_values) if plotted_values else 0.0
    ax.set_ylim(0, max_value * 1.22 if max_value > 0 else 1.0)
    return max_value


def plot_results(bandwidth_breakdown_data, avg_ttft_data, plots_dir):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    os.makedirs(plots_dir, exist_ok=True)
    target_dir = plots_dir

    setup_style(plt)

    engine_styles = {
        "sdma": {"label": "SDMA", "hatch": "", "alpha": 1.0},
        "mte": {"label": "MTE", "hatch": "//", "alpha": 0.9},
    }
    legend_handles = [
        Patch(facecolor="white", edgecolor="black", label="SDMA"),
        Patch(facecolor="white", edgecolor="black", hatch="//", label="MTE"),
    ]

    fig_left, axes_left = plt.subplots(1, 2, figsize=(6, 8), sharey=True)
    bandwidth_max_values = []

    for ax, (transfer_type, title) in zip(axes_left, TRANSFER_TYPES):
        max_value = plot_grouped_bars(
            ax=ax,
            values_by_model=bandwidth_breakdown_data[transfer_type],
            ylabel="",
            title=title,
            model_styles=LEFT_MODEL_STYLES,
            engine_styles=engine_styles,
        )
        ax.axhline(
            y=MEASURED_HARDWARE_BW_LIMIT_GBPS,
            color="gray",
            linestyle=":",
            linewidth=2.0,
        )
        bandwidth_max_values.append(max_value)

    shared_bandwidth_limit = max(
        max(bandwidth_max_values) * 1.22 if bandwidth_max_values else 1.0,
        MEASURED_HARDWARE_BW_LIMIT_GBPS * 1.05,
    )
    for ax in axes_left:
        ax.set_ylim(0, shared_bandwidth_limit)

    fig_left.supylabel(
        BANDWIDTH_YLABEL,
        fontsize=STYLE["shared_ylabel_size"],
        x=STYLE["shared_ylabel_x"],
    )

    axes_left[1].legend(
        handles=legend_handles,
        loc="upper right",
        frameon=True,
    )
    save_figure(fig_left, "motiv-max-bandwidth-comparison", target_dir)
    plt.close(fig_left)

    fig_right, ax_right = plt.subplots(figsize=(3.5, 8))
    plot_grouped_bars(
        ax=ax_right,
        values_by_model=avg_ttft_data,
        ylabel=AVG_TTFT_YLABEL,
        title="",
        model_styles=RIGHT_MODEL_STYLES,
        engine_styles=engine_styles,
        ylabel_size=STYLE["shared_ylabel_size"],
        ylabel_pad=STYLE["axis_label_pad"],
    )
    ax_right.legend(
        handles=legend_handles,
        loc="upper right",
        frameon=True,
    )
    save_figure(fig_right, "motiv-mte-contention-ttft", target_dir)
    plt.close(fig_right)

    return target_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Plot manual SDMA/MTE motivation figures.")
    default_plots_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "perf_plots_manual")
    parser.add_argument("--plots-dir", default=default_plots_dir, help="output plots directory")
    return parser.parse_args()


def main():
    args = parse_args()
    validate_bar_pair_data(BANDWIDTH_BREAKDOWN_DATA, "bandwidth data")
    validate_avg_ttft_data(AVG_TTFT_DATA)
    out_dir = plot_results(BANDWIDTH_BREAKDOWN_DATA, AVG_TTFT_DATA, args.plots_dir)
    print(f"Plot output directory: {out_dir}")


if __name__ == "__main__":
    main()
