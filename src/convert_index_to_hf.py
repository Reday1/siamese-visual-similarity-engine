"""
Adds Hugging Face URLs to artifacts/image_index.json.

Preserves entry order exactly (embeddings.npy positions depend on it).
Only adds a "url" field to each entry — does not touch "path" or "species".

Usage (from project root):
    python src/convert_index_to_hf.py

Before running, set HF_DATASET below to your actual dataset ID.
"""

import json
from pathlib import Path

# ── Configuration ────────────────────────────────────────────────────────
# Replace with your actual Hugging Face dataset identifier
HF_DATASET = "Reday1/cub-200-2011"

HF_BASE_URL = f"https://huggingface.co/datasets/{HF_DATASET}/resolve/main/"

# ── Paths ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
INDEX_PATH = ROOT / "artifacts" / "image_index.json"

# ── Load ─────────────────────────────────────────────────────────────────
with open(INDEX_PATH, "r") as f:
    index = json.load(f)

print(f"Loaded {len(index)} entries from {INDEX_PATH.name}")

# ── Add URLs (order-preserving) ──────────────────────────────────────────
for item in index:
    item["url"] = HF_BASE_URL + item["path"]

# ── Save ─────────────────────────────────────────────────────────────────
with open(INDEX_PATH, "w") as f:
    json.dump(index, f, indent=4)

print(f"Done — added 'url' field to all {len(index)} entries.")
print(f"Base URL: {HF_BASE_URL}")
print(f"Sample:   {index[0]['url']}")
