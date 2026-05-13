import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator


def main():
    plt.rcParams["font.family"] = "Times New Roman"

    # Use evenly spaced bar-group positions so the layout stays compact
    # even when categories are added or removed.
    x_labels = np.array([5, 10, 15, 20, 25], dtype=float)
    x = np.arange(len(x_labels), dtype=float) * 4.0
    Ours = np.array([1.00, 0.82, 0.71, 0.63, 0.57])
    mc_RDMA = np.array([1.49, 1.22, 1.01, 0.88, 0.80])
    Ours_SDMA = np.array([1.23, 1.03, 0.88, 0.76, 0.67])
    width = 1.0
    offsets = np.array([-1, 0, 1]) * (width + 0.25)

    fig, ax = plt.subplots(figsize=(7.4, 3.8), dpi=100)

    ax.bar(
        x + offsets[0],
        mc_RDMA,
        width=width,
        color="#2A6F84",
        edgecolor="#1F5362",
        linewidth=0.9,
        label="RDMA",
        zorder=3,
    )
    ax.bar(
        x + offsets[1],
        Ours_SDMA,
        width=width,
        color="#F3B36C",
        edgecolor="#C88A44",
        linewidth=0.9,
        label="Basic UB",
        zorder=3,
    )
    ax.bar(
        x + offsets[2],
        Ours,
        width=width,
        color="#FF7F0E",
        edgecolor="#C96509",
        linewidth=0.9,
        label="Ours",
        zorder=3,
    )
    left_edge = x[0] + offsets[0] - width / 2
    right_edge = x[-1] + offsets[-1] + width / 2
    ax.set_xlim(left_edge - 0.4, right_edge + 0.4)
    ax.set_ylim(0, 1.5)
    ax.set_xlabel("Cache ratio per NPU", fontsize=22)

    ax.set_xticks(x)
    ax.set_xticklabels([str(int(v)) for v in x_labels])
    ax.set_yticks([0.0, 0.5, 1.0, 1.5])
    ax.tick_params(axis="both", labelsize=18, length=0)

    ax.yaxis.set_minor_locator(MultipleLocator(0.25))
    ax.grid(which="both", axis="y", linestyle=":", color="0.6", linewidth=1.0)
    ax.set_axisbelow(True)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.28),
        ncol=3,
        frameon=False,
        fontsize=18,
        handlelength=2.1,
        handletextpad=0.45,
        markerfirst=False,
        columnspacing=1.2,
    )

    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

    fig.text(
        0.06,
        0.5,
        "Normalized Training Time",
        rotation=90,
        va="center",
        ha="center",
        fontsize=22,
    )

    fig.subplots_adjust(left=0.12, right=0.99, bottom=0.20, top=0.82)
    plt.savefig("plot-bar-output.png", dpi=100)
    plt.savefig("plot-bar-output.pdf")
    plt.show()


if __name__ == "__main__":
    main()
