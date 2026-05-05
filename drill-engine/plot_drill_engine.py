#!/usr/bin/env python3
"""Plot a compact bandwidth drill-down figure with six transfer methods."""

from pathlib import Path

import matplotlib.pyplot as plt


OUTPUT_STEM = "drill_engine_bandwidth"

SELECTED_BLOCK_SIZES = [8 * 1024, 32 * 1024, 128 * 1024, 512 * 1024]
BLOCK_LABELS = ["8KB", "32KB", "128KB", "512KB"]

# Fill in the measured bandwidths (unit: GB/s) for each method.
# The four values correspond to 8KB, 32KB, 128KB, and 512KB.
SERIES_DATA_GBPS = {
    "RDMA": [5.1, 17.0, 21.0, 23.0],
    "SDMA": [7.1, 18.0, 61.0, 110.0],
    "MTE": [81.0, 125.0, 160.0, 164.0],
    "MTE-db": [89.0, 131.0, 164.0, 164.0],
    "MTE-proxy": [68.0, 118.0, 140.0, 145.0],
    "MTE-db-proxy": [86.0, 120.0, 160.0, 161.0],
}

METHOD_ORDER = [
    "RDMA",
    "SDMA",
    "MTE",
    "MTE-db",
    "MTE-proxy",
    "MTE-db-proxy",
]

COLORS = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2"]
MARKERS = ["o", "s", "^", "D", "P", "X"]


def setup_style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 16,
            "axes.labelsize": 18,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 11,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def validate_series(series):
    for method in METHOD_ORDER:
        if method not in series:
            raise KeyError(f"Missing method data: {method}")
        values = series[method]
        if len(values) != len(SELECTED_BLOCK_SIZES):
            raise ValueError(
                f"{method} must provide exactly {len(SELECTED_BLOCK_SIZES)} "
                f"values for {BLOCK_LABELS}."
            )


def plot(series):
    setup_style()

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    x_positions = list(range(len(BLOCK_LABELS)))
    max_y = max(max(values) for values in series.values())

    for idx, method in enumerate(METHOD_ORDER):
        y_values = series[method]
        color = COLORS[idx % len(COLORS)]
        marker = MARKERS[idx % len(MARKERS)]
        ax.plot(
            x_positions,
            y_values,
            color=color,
            marker=marker,
            linewidth=2.4,
            markersize=7.0,
            label=method,
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(BLOCK_LABELS)
    ax.set_xlabel("Message Size")
    ax.set_ylabel("Bandwidth (GB/s)")
    ax.set_ylim(0, max_y * 1.32 if max_y > 0 else 1.0)
    ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.35)
    ax.set_axisbelow(True)
    ax.legend(
        loc="upper left",
        ncol=2,
        frameon=True,
        columnspacing=0.8,
        handlelength=1.8,
    )

    fig.tight_layout()
    return fig


def save_figure(fig):
    output_dir = Path(__file__).resolve().parent
    pdf_path = output_dir / f"{OUTPUT_STEM}.pdf"
    png_path = output_dir / f"{OUTPUT_STEM}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to: {pdf_path}")
    print(f"Saved figure to: {png_path}")


def main():
    series = SERIES_DATA_GBPS
    validate_series(series)
    fig = plot(series)
    save_figure(fig)


if __name__ == "__main__":
    main()
