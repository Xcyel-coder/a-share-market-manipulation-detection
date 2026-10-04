"""Expert activation analysis + ablation study + visualization."""
import sys
import json
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "model"))

from loguru import logger
logger.remove()
logger.add(sys.stderr, level="INFO")

from data.astock_dataset import AStockFinetuneDataset, build_cv_splits
from models.finetune_model import FineTuneModel
from training.finetune_trainer import FineTuneTrainer
from analysis.expert_analysis import collect_expert_activations, ablation_study, save_analysis
from torch.utils.data import DataLoader

BASE = Path(__file__).parent
CASES_DIR = str(BASE / "data_pipeline" / "data" / "processed" / "extracted_cases")
ASTOCK_DIR = str(BASE / "data_pipeline" / "data" / "raw" / "astock")
OUTPUT_DIR = BASE / "model" / "outputs" / "finetune"

MODEL_CFG = {
    "d_model": 256, "n_heads": 8, "n_layers": 6, "d_ff": 512,
    "dropout": 0.1, "max_seq_len": 512,
    "moe": {"num_routed_experts": 8, "num_shared_experts": 2, "top_k": 2,
            "router_jitter": 0.0, "load_balance_weight": 0.001},
}
FEAT_CFG = {"returns_horizons": [1, 5, 10, 20, 60], "volume_ma_window": 20,
            "rsi_period": 14, "bb_period": 20, "atr_period": 14}


def load_cases():
    cases = []
    for f in sorted(Path(CASES_DIR).glob("*_extracted.json")):
        with open(f, "r", encoding="utf-8") as fh:
            case = json.load(fh)
            if case.get("stock_code") and case.get("manipulation_start_date"):
                cases.append(case)
    return cases


def build_full_dataset(cases):
    return AStockFinetuneDataset(
        cases, ASTOCK_DIR, FEAT_CFG, seq_len=60, stride=5,
        min_bars=30, overlap_ratio=0.5, oversample_positive=1,
        cot_embeddings={},
    )


def evaluate_model(model, loader, device):
    from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            outputs = model(batch_gpu)
            probs = torch.softmax(outputs["logits"], dim=-1)[:, 1]
            preds = (probs > 0.5).long()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["label"].numpy())
            all_probs.extend(probs.cpu().numpy())

    f1 = f1_score(all_labels, all_preds, zero_division=0)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.0
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    return {"f1": f1, "auc": auc, "precision": prec, "recall": rec}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    cases = load_cases()
    logger.info(f"Loaded {len(cases)} cases")

    ds = build_full_dataset(cases)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
    n_features = ds.get_n_features()

    analysis_dir = OUTPUT_DIR / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    for level in ["L0", "L1", "L2", "L3"]:
        ckpt_path = OUTPUT_DIR / level / "fold_0" / "best.pt"
        if not ckpt_path.exists():
            logger.warning(f"No checkpoint for {level}, skipping")
            continue

        logger.info(f"\n{'='*50}")
        logger.info(f"Analyzing {level} (fold 0 best model)")
        logger.info(f"{'='*50}")

        level_cfg = {}
        if level in ("L2", "L3"):
            level_cfg = {"custom_router": True, "metadata_dim": 8}
        if level == "L3":
            level_cfg["expert_freeze"] = True
            level_cfg["freeze_top_n_experts"] = 4

        model = FineTuneModel.from_pretrained(
            "", MODEL_CFG, n_features=n_features,
            level=level, level_cfg=level_cfg,
            cot_cfg={"embedding_dim": 384, "projection_dim": 128},
        )

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        model.to(device)
        model.eval()

        logger.info("Collecting expert activations...")
        stats = collect_expert_activations(model, loader, device)

        logger.info("Running ablation study...")

        def model_factory(lv=level, lv_cfg=level_cfg, ckpt_data=ckpt):
            m = FineTuneModel.from_pretrained(
                "", MODEL_CFG, n_features=n_features,
                level=lv, level_cfg=lv_cfg,
                cot_cfg={"embedding_dim": 384, "projection_dim": 128},
            )
            m.load_state_dict(ckpt_data["model_state_dict"], strict=False)
            return m

        ablation = ablation_study(model_factory, loader, device, evaluate_model)

        level_dir = analysis_dir / level
        level_dir.mkdir(parents=True, exist_ok=True)
        save_analysis(stats, ablation, str(level_dir))

        logger.info(f"{level} analysis complete")

    generate_visualization(analysis_dir)
    logger.info("\nAll analysis complete!")


def generate_visualization(analysis_dir):
    """Generate matplotlib visualizations."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping visualization")
        return

    plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    for level in ["L0", "L1", "L2", "L3"]:
        act_file = analysis_dir / level / "expert_activations.json"
        abl_file = analysis_dir / level / "ablation_results.json"
        if not act_file.exists():
            continue

        with open(act_file) as f:
            stats = json.load(f)
        with open(abl_file) as f:
            ablation = json.load(f)

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f"{level} Expert Analysis", fontsize=16, fontweight="bold")

        pos_data = np.array(stats["counts_pos"])
        neg_data = np.array(stats["counts_neg"])

        im = axes[0, 0].imshow(pos_data, cmap="YlOrRd", aspect="auto")
        axes[0, 0].set_title("Positive (Manipulation) Expert Activation")
        axes[0, 0].set_xlabel("Expert ID")
        axes[0, 0].set_ylabel("Layer")
        plt.colorbar(im, ax=axes[0, 0])

        im2 = axes[0, 1].imshow(neg_data, cmap="YlGnBu", aspect="auto")
        axes[0, 1].set_title("Negative (Normal) Expert Activation")
        axes[0, 1].set_xlabel("Expert ID")
        axes[0, 1].set_ylabel("Layer")
        plt.colorbar(im2, ax=axes[0, 1])

        diff = pos_data - neg_data
        im3 = axes[1, 0].imshow(diff, cmap="RdBu_r", aspect="auto",
                                 vmin=-np.abs(diff).max(), vmax=np.abs(diff).max())
        axes[1, 0].set_title("Activation Difference (Pos - Neg)")
        axes[1, 0].set_xlabel("Expert ID")
        axes[1, 0].set_ylabel("Layer")
        plt.colorbar(im3, ax=axes[1, 0])

        expert_ids = [a["expert_id"] for a in ablation["ablations"]]
        f1_drops = [a["f1_drop"] for a in ablation["ablations"]]
        colors = ["#e74c3c" if d > 0 else "#2ecc71" for d in f1_drops]
        axes[1, 1].bar(range(len(expert_ids)), f1_drops, color=colors,
                        tick_label=[f"E{eid}" for eid in expert_ids])
        axes[1, 1].set_title("Ablation Study: F1 Drop per Expert")
        axes[1, 1].set_xlabel("Expert (sorted by impact)")
        axes[1, 1].set_ylabel("F1 Drop")
        axes[1, 1].axhline(y=0, color="black", linestyle="-", linewidth=0.5)

        plt.tight_layout()
        fig.savefig(analysis_dir / f"{level}_expert_analysis.png", dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved {level}_expert_analysis.png")

    levels_data = {}
    for level in ["L0", "L1", "L2", "L3"]:
        abl_file = analysis_dir / level / "ablation_results.json"
        if abl_file.exists():
            with open(abl_file) as f:
                levels_data[level] = json.load(f)

    if len(levels_data) >= 2:
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(8)
        width = 0.2
        for i, (lv, data) in enumerate(levels_data.items()):
            drops = {a["expert_id"]: a["f1_drop"] for a in data["ablations"]}
            vals = [drops.get(e, 0) for e in range(8)]
            ax.bar(x + i * width, vals, width, label=lv, alpha=0.8)
        ax.set_xlabel("Expert ID")
        ax.set_ylabel("F1 Drop when ablated")
        ax.set_title("Cross-Level Expert Importance Comparison")
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels([f"E{i}" for i in range(8)])
        ax.legend()
        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
        plt.tight_layout()
        fig.savefig(analysis_dir / "cross_level_comparison.png", dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Saved cross_level_comparison.png")


if __name__ == "__main__":
    main()
