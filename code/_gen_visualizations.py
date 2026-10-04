"""Generate Phase 4 visualizations for the thesis."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUTPUT_DIR = Path(__file__).parent / "model" / "outputs" / "phase4"
FIG_DIR = OUTPUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Load results
with open(OUTPUT_DIR / "comparison_results.json", encoding="utf-8") as f:
    comparison = json.load(f)
with open(OUTPUT_DIR / "robustness_results.json", encoding="utf-8") as f:
    robustness = json.load(f)
with open(OUTPUT_DIR / "backtest_results.json", encoding="utf-8") as f:
    backtest = json.load(f)

gemini_path = OUTPUT_DIR / "gemini_zeroshot_results.json"
has_gemini = gemini_path.exists()
if has_gemini:
    with open(gemini_path, encoding="utf-8") as f:
        gemini = json.load(f)
    comparison["Gemini-ZeroShot"] = gemini.get("metrics", {})


# ============================================================
# Figure 1: Comparison bar chart (F1, ROC-AUC, MCC)
# ============================================================
def plot_comparison():
    methods = sorted(comparison.keys(), key=lambda x: comparison[x].get("f1", 0), reverse=True)
    f1s = [comparison[m].get("f1", 0) for m in methods]
    aucs = [comparison[m].get("roc_auc", 0) for m in methods]
    mccs = [comparison[m].get("mcc", 0) for m in methods]

    x = np.arange(len(methods))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))
    b1 = ax.bar(x - width, f1s, width, label="F1", color="#2196F3", alpha=0.85)
    b2 = ax.bar(x, aucs, width, label="ROC-AUC", color="#4CAF50", alpha=0.85)
    b3 = ax.bar(x + width, mccs, width, label="MCC", color="#FF9800", alpha=0.85)

    ax.set_ylabel("分数", fontsize=12)
    ax.set_title("方法对比：F1 / ROC-AUC / MCC", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=35, ha="right", fontsize=9)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.08)
    ax.axhline(y=0.99, color="gray", linestyle="--", alpha=0.3)
    ax.grid(axis="y", alpha=0.3)

    for bars in [b1, b2, b3]:
        for bar in bars:
            if bar.get_height() > 0.01:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "comparison_bar.png", dpi=200)
    plt.close()
    print("  comparison_bar.png")


# ============================================================
# Figure 2: Ablation heatmap
# ============================================================
def plot_ablation():
    ablation_items = [
        ("完整模型 (L3)", "MoE-L3"),
        ("去专家冻结 (L2)", "MoE-L2"),
        ("去路由+冻结 (L1)", "MoE-L1"),
        ("去思维链 (L0)", "MoE-L0"),
        ("无预训练", "MoE-NoPretrain"),
    ]

    metrics_names = ["f1", "roc_auc", "mcc", "precision", "recall"]
    display_names = ["F1", "ROC-AUC", "MCC", "精确率", "召回率"]

    data = []
    labels = []
    for name, key in ablation_items:
        if key in comparison:
            row = [comparison[key].get(m, 0) for m in metrics_names]
            data.append(row)
            labels.append(name)

    if not data:
        return

    data = np.array(data)
    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(data, cmap="YlOrRd", aspect="auto", vmin=0.65, vmax=1.0)

    ax.set_xticks(range(len(display_names)))
    ax.set_xticklabels(display_names, fontsize=11)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)

    for i in range(len(labels)):
        for j in range(len(display_names)):
            ax.text(j, i, f"{data[i, j]:.4f}", ha="center", va="center", fontsize=9,
                    color="white" if data[i, j] > 0.9 else "black")

    ax.set_title("消融实验热力图", fontsize=14)
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "ablation_heatmap.png", dpi=200)
    plt.close()
    print("  ablation_heatmap.png")


# ============================================================
# Figure 3: Robustness grouped bars
# ============================================================
def plot_robustness():
    rob_results = robustness.get("results", {})

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    titles = ["时间窗口", "板块类型", "操纵类型", "市值规模"]
    keys = ["time_window", "board_type", "manipulation_type", "market_cap"]
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]

    for idx, (key, title, color) in enumerate(zip(keys, titles, colors)):
        ax = axes[idx // 2][idx % 2]
        groups = rob_results.get(key, {})
        if not groups:
            ax.text(0.5, 0.5, "无数据", ha="center", va="center")
            ax.set_title(title)
            continue

        names = list(groups.keys())
        f1s = [groups[n].get("f1", 0) for n in names]
        aucs = [groups[n].get("roc_auc", 0) for n in names]
        n_cases = [groups[n].get("n_cases", 0) for n in names]

        x = np.arange(len(names))
        w = 0.35
        ax.bar(x - w/2, f1s, w, label="F1", color=color, alpha=0.8)
        ax.bar(x + w/2, aucs, w, label="ROC-AUC", color=color, alpha=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels([f"{n}\n(n={c})" for n, c in zip(names, n_cases)], fontsize=9)
        ax.set_ylim(0.9, 1.02)
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

        for i, (f, a) in enumerate(zip(f1s, aucs)):
            ax.text(i - w/2, f + 0.002, f"{f:.3f}", ha="center", fontsize=8)

    plt.suptitle("分组稳健性测试", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "robustness_grouped.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  robustness_grouped.png")


# ============================================================
# Figure 4: Backtest summary
# ============================================================
def plot_backtest():
    details = backtest.get("details", [])
    if not details:
        print("  backtest: no data")
        return

    warned = [d for d in details if d.get("warned")]
    if not warned:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    losses = [d["loss_if_held_pct"] for d in warned]
    crashes = [d["crash_pct"] for d in warned]
    leads = [d["lead_days"] for d in warned]
    names = [f"{d['stock_code']}" for d in warned]

    ax = axes[0]
    sort_idx = np.argsort(losses)
    sorted_losses = [losses[i] for i in sort_idx]
    sorted_names = [names[i] for i in sort_idx]
    colors_l = ["#E53935" if l < 0 else "#4CAF50" for l in sorted_losses]
    ax.barh(range(len(sorted_losses)), sorted_losses, color=colors_l, alpha=0.8)
    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names, fontsize=7)
    ax.set_xlabel("继续持有的损益 (%)")
    ax.set_title("各案例规避损失情况", fontsize=12)
    ax.axvline(x=0, color="black", linewidth=0.5)
    ax.grid(axis="x", alpha=0.3)

    ax = axes[1]
    ax.scatter(leads, [abs(l) for l in losses], c="#2196F3", s=60, alpha=0.7, edgecolors="white")
    ax.set_xlabel("预警提前天数（操纵期结束前）")
    ax.set_ylabel("规避损失绝对值 (%)")
    ax.set_title("预警提前期 vs 损失幅度", fontsize=12)
    ax.grid(alpha=0.3)

    plt.suptitle(f"回测：{len(warned)}/{backtest['total_cases']} 案例预警，"
                 f"平均规避损失：{backtest['avg_avoided_loss_pct']:.1f}%", fontsize=13)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "backtest_summary.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  backtest_summary.png")


# ============================================================
# Figure 5: MoE level progression
# ============================================================
def plot_level_progression():
    levels = ["MoE-L0", "MoE-L1", "MoE-L2", "MoE-L3"]
    level_labels = ["L0\n基线", "L1\n+思维链蒸馏", "L2\n+元数据路由", "L3\n+专家冻结"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax_idx, (metric, name) in enumerate([("f1", "F1 分数"), ("roc_auc", "ROC-AUC"), ("mcc", "MCC")]):
        vals = [comparison.get(l, {}).get(metric, 0) for l in levels]
        stds = [comparison.get(l, {}).get(f"{metric}_std", 0) for l in levels]

        ax = axes[ax_idx]
        bars = ax.bar(range(4), vals, yerr=stds, capsize=5,
                      color=["#90CAF9", "#42A5F5", "#1565C0", "#0D47A1"], alpha=0.85)
        ax.set_xticks(range(4))
        ax.set_xticklabels(level_labels, fontsize=9)
        ax.set_title(name, fontsize=13)
        ax.set_ylim(min(vals) * 0.95, max(vals) * 1.03)
        ax.grid(axis="y", alpha=0.3)

        for i, v in enumerate(vals):
            ax.text(i, v + 0.002, f"{v:.4f}", ha="center", fontsize=9, fontweight="bold")

    plt.suptitle("渐进式微调：L0 → L3 性能演化", fontsize=14)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "level_progression.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  level_progression.png")


if __name__ == "__main__":
    print("Generating Phase 4 visualizations...")
    plot_comparison()
    plot_ablation()
    plot_robustness()
    plot_backtest()
    plot_level_progression()
    print(f"\nAll figures saved to {FIG_DIR}")
