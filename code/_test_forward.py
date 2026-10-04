"""Quick test of model forward pass."""
import sys
import json
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "model"))

from loguru import logger
logger.remove()
logger.add(sys.stderr, level="INFO")

from data.astock_dataset import AStockFinetuneDataset
from models.finetune_model import FineTuneModel
from torch.utils.data import DataLoader

cases_dir = str(Path(__file__).parent / "data_pipeline" / "data" / "processed" / "extracted_cases")
astock_dir = str(Path(__file__).parent / "data_pipeline" / "data" / "raw" / "astock")

cases = []
for f in sorted(Path(cases_dir).glob("*_extracted.json")):
    with open(f, "r", encoding="utf-8") as fh:
        case = json.load(fh)
        if case.get("stock_code") and case.get("manipulation_start_date"):
            cases.append(case)
print(f"Cases: {len(cases)}")

feat_cfg = {"returns_horizons": [1, 5, 10, 20, 60], "volume_ma_window": 20}
ds = AStockFinetuneDataset(cases[:8], astock_dir, feat_cfg, seq_len=60, stride=10, oversample_positive=1)
print(f"Dataset: {len(ds)} windows, n_features: {ds.get_n_features()}")

loader = DataLoader(ds, batch_size=16, shuffle=True)
batch = next(iter(loader))
print(f"Batch features: {batch['features'].shape}, labels: {batch['label'].tolist()}")

model_cfg = {
    "d_model": 256, "n_heads": 8, "n_layers": 6, "d_ff": 512,
    "dropout": 0.1, "max_seq_len": 512,
    "moe": {"num_routed_experts": 8, "num_shared_experts": 2, "top_k": 2,
            "router_jitter": 0.0, "load_balance_weight": 0.001},
}
model = FineTuneModel.from_pretrained("", model_cfg, n_features=ds.get_n_features(), level="L0")
n_params = sum(p.numel() for p in model.parameters())
print(f"Model params: {n_params/1e6:.1f}M")

with torch.no_grad():
    outputs = model(batch)
print(f"Logits shape: {outputs['logits'].shape}")
print(f"Predictions: {outputs['logits'].argmax(dim=-1).tolist()}")
print("Forward pass OK!")
