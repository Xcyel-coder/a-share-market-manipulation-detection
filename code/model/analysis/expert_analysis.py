"""
Expert activation analysis and visualization for Phase 3.

Produces:
1. Expert activation heatmaps (layer × expert)
2. Per-input-type activation distribution (manipulation vs normal)
3. Ablation study: mask each expert and measure F1 drop
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from loguru import logger


def collect_expert_activations(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> dict:
    """
    Run model on data and record which experts activate for each sample.
    Returns per-layer, per-expert activation counts split by label.
    """
    model.eval()
    encoder = model.encoder if hasattr(model, "encoder") else model

    n_layers = len(encoder.blocks)
    n_experts = encoder.blocks[0].moe_ffn.num_routed

    stats = {
        "counts_pos": np.zeros((n_layers, n_experts)),
        "counts_neg": np.zeros((n_layers, n_experts)),
        "total_pos": 0,
        "total_neg": 0,
        "routing_probs_pos": np.zeros((n_layers, n_experts)),
        "routing_probs_neg": np.zeros((n_layers, n_experts)),
    }

    with torch.no_grad():
        for batch in dataloader:
            features = batch["features"].to(device)
            labels = batch["label"].numpy()
            B, S, _ = features.shape

            x = encoder.input_proj(features)
            x = encoder.input_norm(x)
            x = x + encoder.pos_embedding[:, :S, :]
            cls_tokens = encoder.cls_token.expand(B, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)

            for layer_idx, block in enumerate(encoder.blocks):
                residual = x
                x_norm = block.norm1(x)
                x_attn, _ = block.attn(x_norm, x_norm, x_norm)
                x = block.dropout(x_attn) + residual

                residual = x
                x_norm = block.norm2(x)

                router = block.moe_ffn.router
                is_metadata_router = hasattr(router, 'metadata_proj')
                if is_metadata_router:
                    gate_input = torch.cat([x_norm, torch.zeros_like(x_norm)], dim=-1)
                    logits = router.gate(gate_input)
                    probs = torch.softmax(logits, dim=-1)
                    _, indices, _ = router(x_norm, metadata=None)
                else:
                    logits = router.gate(x_norm)
                    probs = torch.softmax(logits, dim=-1)
                    _, indices, _ = router(x_norm)

                for b_idx in range(B):
                    lbl = labels[b_idx]
                    key_counts = "counts_pos" if lbl == 1 else "counts_neg"
                    key_probs = "routing_probs_pos" if lbl == 1 else "routing_probs_neg"

                    for e_id in range(n_experts):
                        count = (indices[b_idx] == e_id).float().sum().item()
                        stats[key_counts][layer_idx, e_id] += count
                        stats[key_probs][layer_idx, e_id] += probs[b_idx, :, e_id].mean().item()

                    if layer_idx == 0:
                        if lbl == 1:
                            stats["total_pos"] += 1
                        else:
                            stats["total_neg"] += 1

                ffn_out, _ = block.moe_ffn(x_norm)
                x = block.dropout(ffn_out) + residual

            x = encoder.final_norm(x)

    for key in ["counts_pos", "counts_neg", "routing_probs_pos", "routing_probs_neg"]:
        denom = stats["total_pos"] if "pos" in key else stats["total_neg"]
        if denom > 0:
            stats[key] = stats[key] / denom

    return stats


def ablation_study(
    model_factory,
    val_loader: DataLoader,
    device: torch.device,
    evaluate_fn,
) -> dict:
    """
    Ablation: zero-out each routed expert, measure F1 drop.
    model_factory: callable that returns a fresh model copy.
    evaluate_fn: callable(model, loader, device) -> dict with "f1" key.
    """
    baseline_model = model_factory()
    baseline_model.to(device)
    baseline_metrics = evaluate_fn(baseline_model, val_loader, device)
    baseline_f1 = baseline_metrics["f1"]
    logger.info(f"Ablation baseline F1: {baseline_f1:.4f}")

    encoder = baseline_model.encoder if hasattr(baseline_model, "encoder") else baseline_model
    n_layers = len(encoder.blocks)
    n_experts = encoder.blocks[0].moe_ffn.num_routed

    results = {"baseline_f1": baseline_f1, "ablations": []}

    for e_id in range(n_experts):
        model = model_factory()
        model.to(device)
        enc = model.encoder if hasattr(model, "encoder") else model

        for block in enc.blocks:
            expert = block.moe_ffn.routed_experts[e_id]
            for p in expert.parameters():
                p.data.zero_()

        metrics = evaluate_fn(model, val_loader, device)
        drop = baseline_f1 - metrics["f1"]
        results["ablations"].append({
            "expert_id": e_id,
            "f1_without": metrics["f1"],
            "f1_drop": drop,
        })
        logger.info(f"Expert {e_id} ablated: F1={metrics['f1']:.4f} (drop={drop:+.4f})")

    results["ablations"].sort(key=lambda x: x["f1_drop"], reverse=True)
    return results


def save_analysis(stats: dict, ablation: dict, output_dir: str):
    """Save analysis results as JSON for later visualization."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for key in stats:
        if isinstance(stats[key], np.ndarray):
            stats[key] = stats[key].tolist()

    with open(out / "expert_activations.json", "w") as f:
        json.dump(stats, f, indent=2)

    with open(out / "ablation_results.json", "w") as f:
        json.dump(ablation, f, indent=2)

    logger.info(f"Analysis saved to {out}")
