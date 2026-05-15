import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


def annotate_bars(ax, bars, fmt, dy, color="#9C3D1E", fontsize=12):
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + dy,
            fmt(height),
            ha="center",
            va="bottom",
            fontsize=fontsize,
            color=color,
            fontweight="bold",
        )


def main():
    labels = ["HB", "TA", "TA+Mig"]
    x = np.array([2.3, 2.4, 2.5], dtype=float)
    width = 0.02

    values = np.array([1.00, 0.81, 0.72])

    fig, ax = plt.subplots(figsize=(7.4, 3.8), dpi=100)

    bar_style = {
        "edgecolor": "#555555",
        "linewidth": 0.9,
        "zorder": 3,
    }

    bars = ax.bar(
        x,
        values,
        width=width,
        color=["#4CAF50", "#4F81BD", "#E74C3C"],
        **bar_style,
    )
    ax.set_xlim(2.28, 2.52)
    ax.set_ylabel("Norm. fetch latency", fontsize=22)
    ax.set_ylim(0.0, 1.35)
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.tick_params(axis="x", labelsize=22, length=0)
    ax.tick_params(axis="y", labelsize=20, length=0)
    ax.grid(which="major", axis="y", linestyle=":", color="0.60", linewidth=1.0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        spine.set_color("#333333")

    annotate_bars(ax, bars, lambda h: "base" if abs(h - 1.0) < 1e-6 else f"{(h - 1.0) * 100:+.0f}%", 0.03, fontsize=15)

    ax.legend(
        handles=[
            Patch(facecolor="#4CAF50", edgecolor="#2E7D32", linewidth=0.9),
            Patch(facecolor="#4F81BD", edgecolor="#2F5D91", linewidth=0.9),
            Patch(facecolor="#E74C3C", edgecolor="#B73A2E", linewidth=0.9),
        ],
        labels=["Hash-based", "Topology-aware", "Aware + migration"],
        loc="upper right",
        bbox_to_anchor=(1.0, 1.05),
        ncol=1,
        frameon=False,
        fontsize=18,
        handlelength=2.2,
        handletextpad=0.5,
        labelspacing=0.45,
        markerfirst=False,
    )

    fig.subplots_adjust(left=0.18, right=0.98, top=0.96, bottom=0.19)
    fig.savefig("topology_aware_placement_drilldown.png", dpi=100)
    fig.savefig("topology_aware_placement_drilldown.pdf")
    plt.show()


if __name__ == "__main__":
    main()
