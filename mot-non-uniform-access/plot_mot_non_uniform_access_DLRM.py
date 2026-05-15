import matplotlib.pyplot as plt

# 提取你提供的 JSON 数据中的核心绘图指标
ratios = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
scale_factor = 50

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

remote_npu_avg_ms = [x * scale_factor for x in [12.3829, 11.9248, 11.5574, 11.2515, 10.9810, 10.7399, 10.5248, 10.3262, 10.1409, 9.9713, 9.8141, 9.6654, 9.5298]]
dram_offnode_avg_ms = [x * scale_factor for x in [13.9599, 13.2271, 12.6464, 12.1687, 11.7541, 11.3876, 11.0670, 10.7732, 10.5018, 10.2560, 10.0325, 9.8229, 9.6364]]
dram_avg_ms = [x * scale_factor for x in [13.854, 13.128, 12.554, 12.083, 11.675, 11.315, 11.001, 10.714, 10.449, 10.210, 9.993, 9.790, 9.610]]
peer_avg_ms = [x * scale_factor for x in [12.218, 11.743, 11.368, 11.061, 10.794, 10.559, 10.353, 10.165, 9.992, 9.836, 9.694, 9.562, 9.444]]
sio_avg_ms = [x * scale_factor for x in [11.614, 11.231, 10.930, 10.683, 10.468, 10.279, 10.113, 9.962, 9.823, 9.698, 9.584, 9.477, 9.382]]

# Random Placement 是明显离群点，单独放在断轴上半部分展示。
random_placement_point = (20, 178.4167 * scale_factor)

fig, (ax_top, ax_bottom) = plt.subplots(
    2,
    1,
    sharex=True,
    figsize=(6.2, 3.2),
    gridspec_kw={"height_ratios": [1.05, 2.45], "hspace": 0.03},
)

series = [
    (dram_offnode_avg_ms, "D", "#7f7f7f", "remoteDRAM (tier-5)"),
    (remote_npu_avg_ms, "v", "#9467bd", "remoteNPU (tier-4)"),
    (dram_avg_ms, "^", "#409240", "localDRAM (tier-3)"),
    (peer_avg_ms, "s", "#d3352f", "peerNPU (tier-2)"),
    (sio_avg_ms, "o", "#2b7bba", "die-to-die (tier-1)"),
]

for ax in (ax_top, ax_bottom):
    for values, marker, color, label in series:
        ax.plot(
            ratios,
            values,
            marker=marker,
            color=color,
            label=label,
            linewidth=2,
            markersize=8,
        )
    ax.scatter(
        random_placement_point[0],
        random_placement_point[1],
        marker="X",
        color="#e67e22",
        s=150,
        label="Random Placement",
        zorder=5,
    )
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.tick_params(labelsize=10)

# 下半部分突出五条接近的折线，上半部分只展示离群点。
ax_bottom.set_ylim(450, 730)
ax_top.set_ylim(8600, 9200)

ax_top.spines["bottom"].set_visible(False)
ax_bottom.spines["top"].set_visible(False)
ax_top.tick_params(labeltop=False, bottom=False)
ax_bottom.xaxis.tick_bottom()

ax_bottom.set_xlim(18, 82)
ax_bottom.set_xticks(range(20, 81, 10))
ax_bottom.set_xlabel("Embedding cache ratio in local HBM (%)", fontsize=12)
fig.text(0.01, 0.50, "Extraction Time (ms)", fontsize=12, rotation=90, va="center", ha="center")

# 绘制断轴标记。
d = 0.012
kwargs = dict(transform=ax_top.transAxes, color="k", clip_on=False, linewidth=1.2)
ax_top.plot((-d, +d), (-d, +d), **kwargs)
ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)

kwargs.update(transform=ax_bottom.transAxes)
ax_bottom.plot((-d, +d), (1 - d, 1 + d), **kwargs)
ax_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

handles, labels = ax_bottom.get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="upper right",
    bbox_to_anchor=(0.78, 0.955),
    ncol=1,
    fontsize=11,
    frameon=True,
    columnspacing=0.6,
    handletextpad=0.55,
    borderpad=0.35,
    labelspacing=0.30,
)

fig.subplots_adjust(left=0.11, right=0.76, top=0.93, bottom=0.16)

fig.savefig("extract_time_chart_with_random_placement.png", dpi=300, bbox_inches="tight")
fig.savefig("extract_time_chart_with_random_placement.pdf", bbox_inches="tight")
plt.show()