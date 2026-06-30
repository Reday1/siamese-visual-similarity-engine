# Siamese Network Image Retrieval — Project README

## Overview
This project trains a Siamese neural network to learn an embedding space where visually/semantically similar images land close together and dissimilar images land far apart — without ever training on a "retrieval" task directly. Once trained, the encoder half of the network is used to embed a full image dataset and power nearest-neighbor image retrieval, evaluated with precision@k and visualized via t-SNE/UMAP and similarity grids.

Two datasets are used in sequence:
1. **Omniglot** — initial proof-of-concept, full pipeline built and validated here first
2. **CUB-200-2011** — fine-grained bird species, applied second once the pipeline is proven end to end

## Team split

| | Person A — Pairs & Model | Person B — Index, Retrieval, Eval & Viz |
|---|---|---|
| Owns | Dataset, pair generation, Siamese architecture, training loop, contrastive loss | Embedding index, k-NN retrieval, metrics, t-SNE/UMAP, similarity grids, report |
| Hands off | A trained encoder checkpoint + a fixed `splits.json` | The final demo (CLI/notebook) + evaluation report |
| Key files | `dataset.py`, `pairs.py`, `model.py`, `train.py` | `embed_index.py`, `retrieval.py`, `evaluate.py`, `visualize.py` |

**Handoff contract:** Person A delivers `encoder_best.keras`, `splits.json`, and a written preprocessing spec (image size, resize method, normalization mean/std). Person B must reproduce this preprocessing exactly at inference time — any mismatch silently degrades retrieval quality without throwing an error.

---

## Current status

- [x] Trained once on Omniglot (training PNG generated)
- [x] Uploaded environment/code to Kaggle
- [ ] **Bug:** early stopping and best-checkpoint saving are keyed off *training* loss instead of *validation* loss — needs fix before this run is trustworthy
- [ ] `splits.json` not yet created for Omniglot
- [ ] Distance-histogram sanity check (positive vs. negative pair distances) not yet added to training loop
- [ ] CUB-200 pipeline not started — deferred until Omniglot pipeline is fully working end to end

**Working order:** finish the entire pipeline (training fixes → splits.json → retrain → handoff → Person B's retrieval/eval/viz) on Omniglot first. Only once that runs cleanly end to end does the team move to CUB-200, at which point it's primarily a data/encoder swap rather than new pipeline work.

---

## Person A — Data Pairing + Siamese Model + Training

### 1. Dataset & splits
- Dataset has natural similarity groups (Omniglot: characters within alphabets; CUB-200: bird species)
- Split by *class*, not just by image, where the goal is generalization to unseen classes — otherwise retrieval evaluation later leaks
- Save the split as a static manifest, `splits.json`:
```json
  [
    {"filepath": "path/to/image.png", "class_id": 12, "split": "train"},
    {"filepath": "path/to/image2.png", "class_id": 12, "split": "val"}
  ]
```
  This is the single source of truth for both pair generation (train rows only) and Person B's evaluation (test rows only). Don't regenerate it ad hoc.

### 2. Pair generation (`pairs.py`)
- Positive pairs: random same-class image pairs
- Negative pairs: random different-class pairs (optionally hard negatives — visually similar but different class — once basic training works)
- Balance ratio ~1:1 positive:negative per batch
- Wrap in a custom `Dataset`/`DataLoader` returning `(img1, img2, label)`, where label = 1 (similar) / 0 (dissimilar)
- Pull only from `splits.json`'s `train` rows

### 3. Architecture (`model.py`)
- Backbone: ResNet18 (pretrained, drop the final FC layer)
- Projection head: `Linear(512→256) → ReLU → Linear(256→128)`
- Single `forward_once(x)` method; the "Siamese" call invokes it twice with shared weights
- L2-normalize the output embedding (keeps cosine distance well-behaved downstream)

### 4. Training (`train.py`)
- Contrastive loss:
