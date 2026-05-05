#!/usr/bin/env python3
"""Plot two bar charts for the UB contention motivation figure.

This script generates a two-panel figure:
- Left: TTFT under standalone / +weight H2D.
- Right: TPOT under standalone / +weight H2D.

The intent is to visualize the motivation in `motivation.tex`:
critical-path collectives on prefill/decode are delayed by background
weight-distribution traffic on the shared UB fabric.

Run:
    python plot_contention_bars.py
"""

from pathlib import Path
from typing import Dict, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np


SCENARIOS = ["standalone", "w/ weight h2d"]
SCENARIO_LABELS = {
    "standalone": "Standalone",
    "w/ weight h2d": "w/ Weight H2D",
}
MODELS = ["Deepseek", "Qwen"]

# Replace these placeholder values with measured numbers.
# Error bars support both symmetric and asymmetric forms:
#   80.0
#   {"lower": 60.0, "upper": 110.0}
ErrorValue = Union[float, Dict[str, float]]

TTFT_MS = {
    "Deepseek": {
        "standalone": 2118.0,
        "w/ weight h2d": 4243.0,
    },
    "Qwen": {
        "standalone": 726.0,
        "w/ weight h2d": 1356.0,
    },
}

TTFT_ERR_MS = {
    "Deepseek": {
        "standalone": {"lower": 5.0, "upper": 9.0},
        "w/ weight h2d": {"lower": 5.0, "upper": 9.0},
    },
    "Qwen": {
        "standalone": {"lower": 5.0, "upper": 9.0},
        "w/ weight h2d": {"lower": 5.0, "upper": 9.0},
    },
}

TPOT_MS = {
    "Deepseek": {
        "standalone": 92.0,
        "w/ weight h2d": 315.0,
    },
    "Qwen": {
        "standalone": 61.0,
        "w/ weight h2d": 168.0,
    },
}

TPOT_ERR_MS = {
    "Deepseek": {
        "standalone": {"lower": 5.0, "upper": 9.0},
        "w/ weight h2d": {"lower": 5.0, "upper": 9.0},
    },
    "Qwen": {
        "standalone": {"lower": 5.0, "upper": 9.0},
        "w/ weight h2d": {"lower": 5.0, "upper": 9.0},
    },
}

COLORS = {
    "standalone": "#4C78A8",
    "w/ weight h2d": "#F58518",
}


def validate_metric(metric_name: str, metric_data: Dict[str, Dict[str, float]]) -> None:
    if set(metric_data) != set(MODELS):
        raise ValueError(f"{metric_name} keys must exactly match MODELS.")
    for model in MODELS:
        if set(metric_data[model]) != set(SCENARIOS):
            raise ValueError(f"{metric_name} scenario keys must match SCENARIOS for {model}.")
        for scenario in SCENARIOS:
            value = metric_data[model][scenario]
            if value <= 0:
                raise ValueError(f"{metric_name} must be positive for {model}/{scenario}.")


def normalize_error_value(error_name: str, model: str, scenario: str, error: ErrorValue) -> Tuple[float, float]:
    if isinstance(error, (int, float)):
        lower = float(error)
        upper = float(error)
    elif isinstance(error, dict):
        if set(error) != {"lower", "upper"}:
            raise ValueError(
                f"{error_name} must use either a number or "
                f"{{'lower': x, 'upper': y}} for {model}/{scenario}."
            )
        lower = float(error["lower"])
        upper = float(error["upper"])
    else:
        raise ValueError(
            f"{error_name} must use either a number or "
            f"{{'lower': x, 'upper': y}} for {model}/{scenario}."
        )
    return lower, upper


def validate_error(
    error_name: str,
    metric_data: Dict[str, Dict[str, float]],
    error_data: Dict[str, Dict[str, ErrorValue]],
) -> None:
    if set(error_data) != set(MODELS):
        raise ValueError(f"{error_name} keys must exactly match MODELS.")
    for model in MODELS:
        if set(error_data[model]) != set(SCENARIOS):
            raise ValueError(f"{error_name} scenario keys must match SCENARIOS for {model}.")
        for scenario in SCENARIOS:
            lower, upper = normalize_error_value(
                error_name, model, scenario, error_data[model][scenario]
            )
            if lower < 0 or upper < 0:
                raise ValueError(f"{error_name} must be non-negative for {model}/{scenario}.")
            if lower > metric_data[model][scenario] or upper > metric_data[model][scenario]:
                raise ValueError(
                    f"{error_name} must not exceed metric value for {model}/{scenario}."
                )


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 20,
            "axes.labelsize": 17,
            "axes.titlesize": 19,
            "xtick.labelsize": 15,
            "ytick.labelsize": 18,
            "legend.fontsize": 17,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def annotate_delta(ax: plt.Axes, x: float, y: float, value: float, baseline: float) -> None:
    if value == baseline:
        label = "base"
    else:
        delta_pct = (value - baseline) / baseline * 100.0
        label = f"+{delta_pct:.0f}%"
    ax.text(
        x,
        y,
        label,
        ha="center",
        va="bottom",
        fontsize=14,
        rotation=0,
    )


def plot_metric(
    ax: plt.Axes,
    metric_data: Dict[str, Dict[str, float]],
    error_data: Dict[str, Dict[str, ErrorValue]],
    ylabel: str,
    title: str,
) -> None:
    x = np.arange(len(MODELS), dtype=float)
    width = 0.28

    max_value = max(
        metric_data[model][scenario]
        + normalize_error_value("plot error", model, scenario, error_data[model][scenario])[1]
        for model in MODELS
        for scenario in SCENARIOS
    )
    annotation_pad = max_value * 0.03

    for idx, scenario in enumerate(SCENARIOS):
        offsets = x + (idx - 0.5) * width
        values = [metric_data[model][scenario] for model in MODELS]
        error_pairs = [
            normalize_error_value("plot error", model, scenario, error_data[model][scenario])
            for model in MODELS
        ]
        lower_errors = [pair[0] for pair in error_pairs]
        upper_errors = [pair[1] for pair in error_pairs]
        bars = ax.bar(
            offsets,
            values,
            width=width,
            color=COLORS[scenario],
            edgecolor="black",
            linewidth=0.8,
            yerr=np.asarray([lower_errors, upper_errors], dtype=float),
            capsize=5,
            error_kw={"elinewidth": 1.2, "capthick": 1.2, "ecolor": "black"},
            label=SCENARIO_LABELS[scenario],
        )
        for bar, model, upper_error in zip(bars, MODELS, upper_errors):
            annotate_delta(
                ax,
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + upper_error + annotation_pad,
                bar.get_height(),
                metric_data[model]["standalone"],
            )

    ax.set_xticks(x)
    ax.set_xticklabels(MODELS)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0.0, max_value * 1.35)
    ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.35)
    ax.set_axisbelow(True)


def save_figure(fig: plt.Figure, output_stem: str, output_dir: Path) -> None:
    pdf_path = output_dir / f"{output_stem}.pdf"
    png_path = output_dir / f"{output_stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to: {pdf_path}")
    print(f"Saved figure to: {png_path}")


def main() -> None:
    output_dir = Path(__file__).resolve().parent
    validate_metric("TTFT_MS", TTFT_MS)
    validate_error("TTFT_ERR_MS", TTFT_MS, TTFT_ERR_MS)
    validate_metric("TPOT_MS", TPOT_MS)
    validate_error("TPOT_ERR_MS", TPOT_MS, TPOT_ERR_MS)
    setup_style()

    fig, (ax_ttft, ax_tpot) = plt.subplots(
        1,
        2,
        figsize=(8, 3),
        constrained_layout=True,
    )

    plot_metric(
        ax_ttft,
        TTFT_MS,
        TTFT_ERR_MS,
        ylabel="P99 TTFT (ms)",
        title=""
    )
    plot_metric(
        ax_tpot,
        TPOT_MS,
        TPOT_ERR_MS,
        ylabel="P99 TPOT (ms)",
        title=""
    )

    handles, labels = ax_ttft.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=True,
        bbox_to_anchor=(0.5, 1.06),
        columnspacing=1.2,
        handlelength=1.4,
    )

    save_figure(fig, "ub-contention", output_dir)


if __name__ == "__main__":
    main()
