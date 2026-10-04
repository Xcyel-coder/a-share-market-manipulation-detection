"""Quick test of A-share data loading pipeline."""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "model"))

from data.astock_dataset import AStockFinetuneDataset, load_case_data, build_cv_splits

cases_dir = Path(__file__).parent / "data_pipeline" / "data" / "processed" / "extracted_cases"
astock_dir = Path(__file__).parent / "data_pipeline" / "data" / "raw" / "astock"

cases = []
for f in sorted(cases_dir.glob("*_extracted.json")):
    with open(f, "r", encoding="utf-8") as fh:
        case = json.load(fh)
        if case.get("stock_code") and case.get("manipulation_start_date"):
            cases.append(case)
print(f"Valid cases: {len(cases)}")

loaded = load_case_data(cases[0], str(astock_dir))
if loaded:
    print(f"Case: {loaded['doc_id']}, bars: {len(loaded['df'])}, "
          f"period: {loaded['manipulation_start']} ~ {loaded['manipulation_end']}")

feat_cfg = {"returns_horizons": [1, 5, 10, 20, 60], "volume_ma_window": 20,
            "rsi_period": 14, "bb_period": 20, "atr_period": 14}

ds = AStockFinetuneDataset(
    cases[:5], str(astock_dir), feat_cfg,
    seq_len=60, stride=5, oversample_positive=3
)
print(f"Dataset: {len(ds)} windows, features: {ds.get_n_features()}")

if len(ds) > 0:
    sample = ds[0]
    print(f"Sample: features={sample['features'].shape}, label={sample['label'].item()}")

splits = build_cv_splits(cases, n_folds=5)
print(f"CV splits: {len(splits)} folds")
for i, (tr, va) in enumerate(splits):
    print(f"  Fold {i}: train={len(tr)}, val={len(va)}")
