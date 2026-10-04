"""
Generate all 5 thesis figures with unified Morandi color palette.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from adjustText import adjust_text

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 200
plt.rcParams["savefig.dpi"] = 200
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["font.size"] = 11

MORANDI = {
    "rose":     "#C4A4A0",
    "sage":     "#9CAF88",
    "blue":     "#7B9EBD",
    "mauve":    "#B5838D",
    "beige":    "#C9B99A",
    "coral":    "#D4A59A",
    "slate":    "#8D99AE",
    "lavender": "#A78BCA",
    "teal":     "#7BACA8",
    "sand":     "#C2B280",
    "plum":     "#8E6C88",
}
M = list(MORANDI.values())

BASE = Path(r"c:\Users\wangm\OneDrive - Audencia\毕业论文")
DATA = BASE / "code" / "model" / "outputs" / "clean_rerun"
OUT = BASE / "figures"

with open(DATA / "comparison_results.json", encoding="utf-8") as f:
    comp = json.load(f)
with open(DATA / "robustness_cv.json", encoding="utf-8") as f:
    robust = json.load(f)
with open(DATA / "backtest_cv.json", encoding="utf-8") as f:
    bt = json.load(f)
with open(DATA / "ablation_results.json", encoding="utf-8") as f:
    ablation = json.load(f)


# ============================================================
# Figure 1: comparison_bar.png
# ============================================================
def fig1_comparison():
    order = [
        ("MoE-L2", "MoE-L2 *"),
        ("MoE-L0", "MoE-L0"),
        ("MoE-L3", "MoE-L3"),
        ("MoE-NoPretrain", "无预训练"),
        ("MoE-L1", "MoE-L1"),
        ("Transformer", "Transformer"),
        ("RandomForest", "随机森林"),
        ("XGBoost", "XGBoost"),
        ("SVM", "SVM"),
        ("LSTM", "LSTM"),
    ]
    labels = [o[1] for o in order]
    f1_vals = [comp[o[0]]["f1"] for o in order]
    auc_vals = [comp[o[0]]["roc_auc"] for o in order]
    mcc_vals = [comp[o[0]]["mcc"] for o in order]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), sharey=True)
    metrics = [("F1 分数", f1_vals, M[2]), ("ROC-AUC", auc_vals, M[1]), ("MCC", mcc_vals, M[3])]

    for ax, (title, vals, color) in zip(axes, metrics):
        colors = [MORANDI["mauve"] if i == 0 else color for i in range(len(vals))]
        ax.barh(range(len(labels)), vals, color=colors, edgecolor="white", height=0.65)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=10)
        ax.set_xlabel(title, fontsize=12, fontweight="bold")
        ax.invert_yaxis()
        if title == "MCC":
            ax.set_xlim(0.5, 1.0)
        else:
            ax.set_xlim(0.85, 1.005)
        for i, v in enumerate(vals):
            ax.text(v + 0.003, i, f"{v:.3f}", va="center", fontsize=8.5, color="#444")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="x", alpha=0.2)

    fig.text(0.5, -0.02, "Gemini-ZeroShot：F1=0.850（因坐标尺度差异未在此显示）",
             ha="center", fontsize=9, style="italic", color="#888")
    fig.suptitle("图 4-1  全方法对比（5 折交叉验证，按 MCC 排序）",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(OUT / "comparison_bar.png", pad_inches=0.2)
    plt.close(fig)
    print("[OK] comparison_bar.png")


# ============================================================
# Figure 2: ablation_heatmap.png — REDESIGNED: radar + waterfall
# ============================================================
def fig2_ablation():
    fig = plt.figure(figsize=(14, 6))
    ax_radar = fig.add_subplot(121, polar=True)
    ax_water = fig.add_subplot(122)

    # --- Left: Radar chart ---
    configs = [
        ("L2（最优）", "w/o Expert Freeze (L2)", MORANDI["mauve"], "-", 2.5),
        ("L0（基线）", "w/o CoT Distill (L0)", MORANDI["blue"], "--", 1.8),
        ("L3（+专家冻结）", "Full Model (L3)", MORANDI["coral"], "-.", 1.8),
        ("L1（+思维链）", "w/o Routing+Freeze (L1)", MORANDI["slate"], ":", 1.8),
        ("无预训练", "w/o Crypto Pretrain", MORANDI["beige"], "--", 1.5),
    ]
    metric_labels = ["F1", "ROC-AUC", "PR-AUC", "MCC", "精确率", "召回率"]
    metric_keys = ["f1", "roc_auc", "pr_auc", "mcc", "precision", "recall"]

    angles = np.linspace(0, 2 * np.pi, len(metric_labels), endpoint=False).tolist()
    angles += angles[:1]

    ax_radar.set_theta_offset(np.pi / 2)
    ax_radar.set_theta_direction(-1)
    ax_radar.set_rlabel_position(30)

    for label, key, color, ls, lw in configs:
        vals = [ablation[key][mk] for mk in metric_keys]
        vals += vals[:1]
        ax_radar.plot(angles, vals, linewidth=lw, linestyle=ls, color=color, label=label)
        ax_radar.fill(angles, vals, alpha=0.06, color=color)

    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(metric_labels, fontsize=9.5)
    ax_radar.set_ylim(0.85, 1.005)
    ax_radar.set_yticks([0.86, 0.90, 0.94, 0.98])
    ax_radar.set_yticklabels(["0.86", "0.90", "0.94", "0.98"], fontsize=7.5, color="#888")
    ax_radar.legend(loc="lower left", bbox_to_anchor=(-0.15, -0.18), fontsize=8.5, ncol=3,
                    framealpha=0.5, edgecolor="#ddd")
    ax_radar.set_title("(a) 多指标雷达图", fontsize=11, fontweight="bold", pad=18)
    ax_radar.grid(alpha=0.25)

    # --- Right: Waterfall chart (sequential: NoPre→L0→L1→L2→L3) ---
    base_mcc = comp["MoE-NoPretrain"]["mcc"]
    steps = [
        ("无预训练\n(基线)", base_mcc, None),
        ("+预训练", comp["MoE-L0"]["mcc"] - base_mcc, "delta"),
        ("+思维链", comp["MoE-L1"]["mcc"] - comp["MoE-L0"]["mcc"], "delta"),
        ("+元数据路由", comp["MoE-L2"]["mcc"] - comp["MoE-L1"]["mcc"], "delta"),
        ("+专家冻结", comp["MoE-L3"]["mcc"] - comp["MoE-L2"]["mcc"], "delta"),
    ]

    cumulative = base_mcc
    bottoms = []
    heights = []
    colors_w = []
    x_labels = []
    for name, val, kind in steps:
        x_labels.append(name)
        if kind is None:
            bottoms.append(0)
            heights.append(val)
            colors_w.append(MORANDI["slate"])
        else:
            if val >= 0:
                bottoms.append(cumulative)
                heights.append(val)
                colors_w.append(MORANDI["sage"])
            else:
                bottoms.append(cumulative + val)
                heights.append(abs(val))
                colors_w.append(MORANDI["coral"])
            cumulative += val

    x_labels.append(f"L3 最终\n({cumulative:.4f})")
    bottoms.append(0)
    heights.append(cumulative)
    colors_w.append(MORANDI["slate"])

    x = np.arange(len(x_labels))
    bars = ax_water.bar(x, heights, bottom=bottoms, color=colors_w, edgecolor="white",
                        width=0.55, zorder=3)

    for i in range(1, len(x_labels) - 1):
        top_prev = bottoms[i - 1] + heights[i - 1] if i > 0 else heights[0]
        ax_water.plot([i - 0.35, i + 0.35], [top_prev, top_prev],
                      color="#bbb", linewidth=0.7, linestyle="--", zorder=2)

    l2_mcc = comp["MoE-L2"]["mcc"]
    l2_cumul = base_mcc + (comp["MoE-L0"]["mcc"] - base_mcc) + \
               (comp["MoE-L1"]["mcc"] - comp["MoE-L0"]["mcc"]) + \
               (comp["MoE-L2"]["mcc"] - comp["MoE-L1"]["mcc"])
    ax_water.plot([2.7, 3.3], [l2_cumul, l2_cumul], color=MORANDI["mauve"], linewidth=2.5, zorder=5)
    ax_water.annotate(f"L2 最优\n{l2_mcc:.4f}", xy=(3.0, l2_cumul),
                      xytext=(3.6, l2_cumul + 0.04), fontsize=9, fontweight="bold",
                      color=MORANDI["mauve"],
                      arrowprops=dict(arrowstyle="->", color=MORANDI["mauve"], lw=1.2))

    for i, (b, h) in enumerate(zip(bottoms, heights)):
        val = h if i == 0 or i == len(x_labels) - 1 else (h if colors_w[i] == MORANDI["sage"] else -h)
        label = f"{val:.4f}" if i == 0 or i == len(x_labels) - 1 else f"{val:+.4f}"
        y_pos = b + h + 0.005 if val >= 0 or i == 0 or i == len(x_labels) - 1 else b - 0.008
        va = "bottom" if val >= 0 or i == 0 or i == len(x_labels) - 1 else "top"
        ax_water.text(i, y_pos, label, ha="center", va=va, fontsize=9.5, color="#444")

    ax_water.set_xticks(x)
    ax_water.set_xticklabels(x_labels, fontsize=9, rotation=15, ha="right")
    ax_water.set_ylabel("MCC", fontsize=11, fontweight="bold")
    ax_water.set_ylim(0.80, 0.96)
    ax_water.set_title("(b) MCC 瀑布图分解", fontsize=11, fontweight="bold")
    ax_water.spines["top"].set_visible(False)
    ax_water.spines["right"].set_visible(False)
    ax_water.grid(axis="y", alpha=0.15)

    legend_elements = [
        mpatches.Patch(facecolor=MORANDI["sage"], edgecolor="white", label="正向贡献"),
        mpatches.Patch(facecolor=MORANDI["coral"], edgecolor="white", label="负向贡献"),
        mpatches.Patch(facecolor=MORANDI["mauve"], edgecolor="white", label="总值"),
    ]
    ax_water.legend(handles=legend_elements, fontsize=8.5, loc="upper left", framealpha=0.5)

    fig.suptitle("图 4-2  消融实验分析",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(OUT / "ablation_heatmap.png", pad_inches=0.25)
    plt.close(fig)
    print("[OK] ablation_heatmap.png")


# ============================================================
# Figure 3: robustness_grouped.png — fixed legends & labels
# ============================================================
def fig3_robustness():
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    dims = [
        ("时间窗口", robust["time_window"], ["2014-2018", "2019-2024"]),
        ("板块类型", robust["board_type"], ["沪市主板", "深市主板", "创业板"]),
        ("违规类型", robust["manipulation_type"], ["连续交易/对倒", "内幕交易", "其他操纵", "其他"]),
    ]

    mcc_color = MORANDI["mauve"]
    f1_color = MORANDI["blue"]

    for ax, (title, data, groups) in zip(axes, dims):
        mcc_vals = [data[g]["mcc"] for g in groups]
        f1_vals = [data[g]["f1"] for g in groups]
        n_cases = [data[g]["n_cases"] for g in groups]

        x = np.arange(len(groups))
        w = 0.32
        ax.bar(x - w / 2, mcc_vals, w, label="MCC", color=mcc_color, edgecolor="white", alpha=0.85)
        ax.bar(x + w / 2, f1_vals, w, label="F1", color=f1_color, edgecolor="white", alpha=0.65)

        short_names = {
            "连续交易/对倒": "连续交易\n/对倒",
            "沪市主板": "沪市\n主板",
            "深市主板": "深市\n主板",
        }
        short_labels = [f"{short_names.get(g, g)}\n(n={n})" for g, n in zip(groups, n_cases)]
        ax.set_xticks(x)
        ax.set_xticklabels(short_labels, fontsize=8.5)
        ax.set_ylim(0.75, 1.08)
        ax.set_ylabel("分数" if ax == axes[0] else "")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.2)

        for i, v in enumerate(mcc_vals):
            ax.text(i - w / 2, v + 0.012, f"{v:.3f}", ha="center", fontsize=7.5,
                    fontweight="bold", color=MORANDI["plum"])
        for i, v in enumerate(f1_vals):
            ax.text(i + w / 2, v + 0.012, f"{v:.3f}", ha="center", fontsize=7.5, color=MORANDI["slate"])

    handles = [
        mpatches.Patch(facecolor=mcc_color, alpha=0.85, edgecolor="white", label="MCC"),
        mpatches.Patch(facecolor=f1_color, alpha=0.65, edgecolor="white", label="F1"),
    ]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.98, 0.97),
               fontsize=10, framealpha=0.5, edgecolor="#ddd")

    fig.suptitle("图 4-3  稳健性测试（5 折验证集汇总，MoE-L2，无数据泄漏）",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(OUT / "robustness_grouped.png", pad_inches=0.2)
    plt.close(fig)
    print("[OK] robustness_grouped.png")


# ============================================================
# Figure 4: backtest_summary.png — fixed label overlap
# ============================================================
def fig4_backtest():
    details = bt["details"]
    warned = [d for d in details if d.get("warned")]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5),
                                    gridspec_kw={"width_ratios": [1, 1.3]})

    names = []
    losses = []
    for d in warned:
        name = d.get("stock_name", d["stock_code"])
        if len(name) > 5:
            name = name[:5] + ".."
        names.append(name)
        losses.append(d["loss_if_held_pct"])

    y = np.arange(len(names))
    colors_loss = [MORANDI["sage"] if v < 0 else MORANDI["coral"] for v in losses]
    ax1.barh(y, losses, color=colors_loss, edgecolor="white", height=0.6)
    ax1.axvline(x=0, color="#888", linewidth=0.8, linestyle="--")
    ax1.set_yticks(y)
    ax1.set_yticklabels(names, fontsize=8.5)
    ax1.invert_yaxis()
    ax1.set_xlabel("继续持有的损益 (%)", fontsize=11)
    ax1.set_title("(a) 预警后继续持有的损益", fontsize=11, fontweight="bold")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.grid(axis="x", alpha=0.2)

    for i, v in enumerate(losses):
        offset = 3 if v >= 0 else -3
        ha = "left" if v >= 0 else "right"
        ax1.text(v + offset, i, f"{v:+.1f}%", va="center", ha=ha, fontsize=7, color="#555")

    lead_days = [d["lead_days"] for d in warned]
    crash_pcts = [abs(d["crash_pct"]) for d in warned]
    sizes = [max(abs(d["loss_if_held_pct"]) * 2.5, 40) for d in warned]
    point_colors = [MORANDI["mauve"] if d["loss_if_held_pct"] < 0 else MORANDI["beige"] for d in warned]

    ax2.scatter(lead_days, crash_pcts, s=sizes, c=point_colors,
                alpha=0.75, edgecolors="white", linewidth=0.8, zorder=3)

    texts = []
    for d in warned:
        name = d.get("stock_name", d["stock_code"])
        if len(name) > 4:
            name = name[:4]
        t = ax2.text(d["lead_days"], abs(d["crash_pct"]), name,
                     fontsize=7, alpha=0.8, zorder=4)
        texts.append(t)

    try:
        adjust_text(texts, ax=ax2, arrowprops=dict(arrowstyle="-", color="#ccc", lw=0.5),
                    expand=(1.4, 1.6), force_text=(0.8, 1.0))
    except Exception:
        pass

    median_lead = np.median(lead_days)
    ax2.axvline(x=median_lead, color=MORANDI["slate"], linewidth=1, linestyle=":", alpha=0.6)
    ax2.text(median_lead + 20, ax2.get_ylim()[1] * 0.95,
             f"中位数\n{median_lead:.0f} 天", ha="left", fontsize=8, color=MORANDI["slate"])

    ax2.set_xlabel("操纵期结束前的预警提前天数", fontsize=11)
    ax2.set_ylabel("操纵后股价跌幅 (%)", fontsize=11)
    ax2.set_title("(b) 预警提前期 vs 跌幅严重程度", fontsize=11, fontweight="bold")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.grid(alpha=0.15)

    summary_text = (f"预警覆盖率：{bt['cases_warned']}/{bt['total_cases']} "
                    f"({bt['warning_rate'] * 100:.1f}%)\n"
                    f"平均规避损失：{bt['avg_avoided_loss_pct']:.1f}%\n"
                    f"平均跌幅：{bt['avg_crash_after_manip_pct']:.1f}%")
    ax2.text(0.98, 0.03, summary_text, transform=ax2.transAxes, fontsize=8.5,
             va="bottom", ha="right",
             bbox=dict(boxstyle="round,pad=0.4", facecolor=MORANDI["beige"], alpha=0.35))

    fig.suptitle("图 4-4  回测验证（5 折验证集汇总，MoE-L2）",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(OUT / "backtest_summary.png", pad_inches=0.2)
    plt.close(fig)
    print("[OK] backtest_summary.png")


# ============================================================
# Figure 5: level_progression.png — cleaner error bars
# ============================================================
def fig5_level_progression():
    levels = ["L0\n(基线)", "L1\n(+思维链)", "L2\n(+元数据路由)", "L3\n(+专家冻结)"]
    keys = ["MoE-L0", "MoE-L1", "MoE-L2", "MoE-L3"]

    mcc = [comp[k]["mcc"] for k in keys]
    mcc_std = [comp[k]["mcc_std"] for k in keys]
    f1 = [comp[k]["f1"] for k in keys]
    f1_std = [comp[k]["f1_std"] for k in keys]
    auc = [comp[k]["roc_auc"] for k in keys]
    auc_std = [comp[k]["roc_auc_std"] for k in keys]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8.5), gridspec_kw={"height_ratios": [1.8, 1]})
    fig.subplots_adjust(hspace=0.45)

    x = np.arange(4)

    eb_kw = dict(capsize=4, capthick=1.2, elinewidth=1)
    ax1.errorbar(x, mcc, yerr=mcc_std, marker="o", markersize=9, linewidth=2.5,
                 color=MORANDI["mauve"], ecolor=MORANDI["rose"], label="MCC", zorder=3, **eb_kw)
    ax1.errorbar(x, f1, yerr=f1_std, marker="s", markersize=7, linewidth=1.8,
                 color=MORANDI["blue"], ecolor=MORANDI["slate"], label="F1", zorder=2, **eb_kw)
    ax1.errorbar(x, auc, yerr=auc_std, marker="^", markersize=7, linewidth=1.8,
                 color=MORANDI["sage"], ecolor=MORANDI["beige"], label="ROC-AUC", zorder=2, **eb_kw)

    best_idx = np.argmax(mcc)
    ax1.scatter([best_idx], [mcc[best_idx]], s=180, facecolors="none",
                edgecolors=MORANDI["mauve"], linewidth=2.5, zorder=4)
    ax1.annotate(f"最优 MCC={mcc[best_idx]:.4f}", (best_idx, mcc[best_idx]),
                 textcoords="offset points", xytext=(15, 12), fontsize=10, fontweight="bold",
                 color=MORANDI["mauve"],
                 arrowprops=dict(arrowstyle="->", color=MORANDI["mauve"], lw=1.5))

    np_mcc = comp["MoE-NoPretrain"]["mcc"]
    ax1.axhline(y=np_mcc, color=MORANDI["slate"], linestyle="--", alpha=0.4, linewidth=1)
    ax1.text(3.5, np_mcc + 0.004, f"无预训练={np_mcc:.4f}", fontsize=8, color=MORANDI["slate"], ha="right")

    ax1.set_xticks(x)
    ax1.set_xticklabels(levels, fontsize=10)
    ax1.set_ylabel("分数（5 折 CV 均值 ± 标准差）", fontsize=10)
    ax1.set_ylim(0.83, 1.015)
    ax1.legend(fontsize=9.5, loc="lower left", framealpha=0.5)
    ax1.set_title("(a) 各层级性能对比", fontsize=12, fontweight="bold")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.grid(alpha=0.15)

    components = ["预训练\n(L0-无预训练)", "思维链\n(L1-L0)", "元数据路由\n(L2-L0)", "专家冻结\n(L3-L2)"]
    deltas = [
        comp["MoE-L0"]["mcc"] - comp["MoE-NoPretrain"]["mcc"],
        comp["MoE-L1"]["mcc"] - comp["MoE-L0"]["mcc"],
        comp["MoE-L2"]["mcc"] - comp["MoE-L0"]["mcc"],
        comp["MoE-L3"]["mcc"] - comp["MoE-L2"]["mcc"],
    ]
    colors = [MORANDI["sage"] if d > 0 else MORANDI["coral"] for d in deltas]
    ax2.bar(range(4), deltas, color=colors, edgecolor="white", width=0.55, zorder=3)
    ax2.axhline(y=0, color="#888", linewidth=0.8)
    ax2.set_xticks(range(4))
    ax2.set_xticklabels(components, fontsize=9)
    ax2.set_ylabel("ΔMCC", fontsize=11)
    ax2.set_title("(b) 各组件对 MCC 的贡献", fontsize=12, fontweight="bold", pad=12)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.grid(axis="y", alpha=0.15)
    ax2.set_ylim(-0.038, 0.035)

    for i, v in enumerate(deltas):
        y_pos = v + 0.002 if v >= 0 else v - 0.002
        va = "bottom" if v >= 0 else "top"
        ax2.text(i, y_pos, f"{v:+.4f}", ha="center", va=va, fontsize=9.5, fontweight="bold", color="#444")

    fig.suptitle("图 4-5  L0-L3 微调消融对比",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(OUT / "level_progression.png", pad_inches=0.2)
    plt.close(fig)
    print("[OK] level_progression.png")


if __name__ == "__main__":
    fig1_comparison()
    fig2_ablation()
    fig3_robustness()
    fig4_backtest()
    fig5_level_progression()
    print(f"\nAll 5 figures saved to {OUT}")
