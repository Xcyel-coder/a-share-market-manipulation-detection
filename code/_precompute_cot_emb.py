"""Pre-compute CoT embeddings and save as .npz cache."""
import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

COT_PATH = Path(__file__).parent / "model" / "outputs" / "cot" / "cot_results.json"
CACHE_PATH = COT_PATH.parent / "cot_embeddings.npz"

print("Loading CoT results...")
with open(COT_PATH, "r", encoding="utf-8") as f:
    cot_results = json.load(f)

case_cots: dict[str, list[str]] = {}
for entry in cot_results:
    doc_id = entry["doc_id"]
    text = entry.get("cot_text", "")
    if text:
        case_cots.setdefault(doc_id, []).append(text)

print(f"Cases with CoT: {len(case_cots)}")

print("Loading embedding model...")
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

embeddings = {}
for i, (doc_id, texts) in enumerate(case_cots.items()):
    combined = "\n\n".join(texts[:4])[:8000]
    emb = model.encode(combined, normalize_embeddings=True)
    embeddings[doc_id] = emb.astype(np.float32)
    if (i + 1) % 10 == 0:
        print(f"  Encoded {i+1}/{len(case_cots)}")

np.savez(CACHE_PATH, **embeddings)
print(f"Saved {len(embeddings)} embeddings to {CACHE_PATH}")
print(f"Embedding dim: {list(embeddings.values())[0].shape}")
