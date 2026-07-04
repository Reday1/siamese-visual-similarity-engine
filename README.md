# Siamese Network Image Retrieval — Project README

## Overview
This project trains a Siamese neural network to learn an embedding space where visually/semantically similar images land close together and dissimilar images land far apart — without ever training on a "retrieval" task directly. Once trained, the encoder half of the network is used to embed a full image dataset and power nearest-neighbor image retrieval, evaluated with precision@k and visualized via t-SNE/UMAP and similarity grids.

Two datasets are used in sequence:
1. **Omniglot** — initial proof-of-concept, full pipeline built and validated here first
2. **CUB-200-2011** — fine-grained bird species, applied second once the pipeline was proven end to end

**Zero-shot framing:** train/val/test splits are made by *species*, not by image. ~40 of 200 species are held out entirely from training and validation. The model is never trained to classify species — it's trained only on same/different pairs via contrastive loss — so evaluation on held-out species tests whether the learned notion of visual similarity generalizes to categories the model has never seen, not whether it memorized 200 known classes.

---

## Team split

| | Person A — Pairs & Model | Person B — Index, Retrieval, Eval & Viz |
|---|---|---|
| Owns | Dataset, pair generation, Siamese architecture, training loop, contrastive loss | Embedding index, k-NN retrieval, metrics, t-SNE/UMAP, similarity grids, report |
| Hands off | A trained encoder checkpoint + a fixed `splits.json` | The final demo (Streamlit app) + evaluation report |
| Key files | `dataset.py`, `pairs.py`, `model.py`, `train.py` | `embed_index.py`, `retrieval.py`, `evaluate.py`, `visualize.py`, `app.py` |

**Handoff contract:** Person A delivers `best_model.weights.h5`, `cub200_splits.json`, and this document's preprocessing spec (below). Person B must reproduce this preprocessing exactly at inference time — any mismatch silently degrades retrieval quality without throwing an error.

**Important — the backbone is fine-tuned, not stock:** `best_model.weights.h5` contains a ResNet50 backbone whose weights have been fine-tuned on CUB-200, starting from ImageNet initialization. It is **not** interchangeable with `ResNet50(weights='imagenet')`. Always load it via `build_encoder()` + `siamese.load_weights(...)`, never by re-downloading stock ImageNet weights and expecting equivalent behavior.

---

## Current status

- [x] Trained on Omniglot (proof of concept, pipeline validated end to end)
- [x] Uploaded environment/code to Kaggle
- [x] CUB-200 pipeline built: species-disjoint splits, pair generation, augmentation, contrastive loss, distributed training
- [x] `cub200_splits.json` generated and **locked** — species assignment to train/val/test is now permanent and must be loaded from this file in every session, never regenerated
- [x] Fixed critical `load_and_resize` bug (was decoding `img1` twice instead of `img2` — caused pair corruption + crashes)
- [x] Fixed Keras 3 optimizer state bug at unfreeze boundary (optimizer must be recreated inside `strategy.scope()` when the trainable-variable set changes)
- [x] Fixed OOM/kernel-restart issue: removed `.cache()` (was caching ~13GB of decoded images in RAM for a dataset that's regenerated every epoch and never reused) and moved `.shuffle()` before decode (was buffering 1000 full decoded image pairs instead of cheap file paths)
- [x] Added corrupt-image handling for CUB-200-2011's known bad JPEG files
- [x] First full 15-epoch training run completed successfully: train/val loss converge cleanly, best checkpoint saved at epoch 14 (val loss ≈ 0.207), no meaningful overfitting until the final epoch
- [ ] **In progress:** re-running with a staged/progressive backbone unfreeze (5 layers unlocked every 3 epochs, 20 layers total across 4 stages, LR stepped down at each stage) instead of unfreezing the entire backbone at once, extended to 20 epochs
- [ ] Real early stopping not yet implemented — current logic only halves LR on plateau (`patience_counter`), it never breaks the loop
- [ ] Validation pairs are still regenerated fresh each epoch rather than fixed — acceptable since species are locked, but final reported metrics should come from a separate, fixed, larger eval pair set (see below), not from the training loop's per-epoch val loss
- [ ] Full embedding index over all test-species images not yet generated
- [ ] k-NN retrieval + precision@k evaluation not yet built
- [ ] t-SNE/UMAP visualization + similarity grids not yet built
- [ ] Streamlit retrieval demo not yet built

**Working order:** finish the staged-unfreeze training run → generate a fixed, larger eval pair set and confirm the checkpoint's true val performance → build the full embedding index → k-NN retrieval + precision@k → t-SNE/UMAP + similarity grids → Streamlit app → final write-up.

---

## Person A — Data Pairing + Siamese Model + Training

### 1. Dataset & splits
- Dataset has natural similarity groups (Omniglot: characters within alphabets; CUB-200: bird species)
- Split by **species**, not by image — the goal is generalization to unseen species, and image-level splits would leak identity information across train/val/test
- Split is saved once to `cub200_splits.json` and **must be loaded from this file in every session** — regenerating the split (e.g. via a fresh `os.listdir()` call on re-extracted data) can silently reassign species across train/val/test between sessions, breaking the zero-shot evaluation without any error being thrown. This was an actual bug encountered during development and is the single most important invariant in this project.
- Format:
```json
{
  "train": {"species_name": ["path1.jpg", "path2.jpg", ...], ...},
  "val":   {"species_name": [...], ...},
  "test":  {"species_name": [...], ...}
}
```

### 2. Pair generation (`pairs.py`)
- Positive pairs: random same-species image pairs
- Negative pairs: mix of random different-species pairs and **online hard negatives** — pairs mined each epoch by embedding a sample of training images with the *current* encoder and selecting the closest cross-species match. Hard-negative ratio ramps from 0.0 (epochs 0–2, warmup) to 0.5 (epoch 3 onward), once an embedding cache exists to mine from.
- Balance ratio ~1:1 positive:negative per class per epoch
- Pull only from the locked split's `train` rows for training pairs; `val`/`test` rows are never seen during training

### 3. Architecture (`model.py`)
- Backbone: ResNet50 (ImageNet-pretrained, `pooling='avg'`, no top)
- Projection head: `Dense(512, relu) → Dropout(0.3) → Dense(256, relu) → Dropout(0.3) → Dense(128)`
- L2-normalized embedding output (custom `L2Normalize` layer — not `Lambda`, for clean serialization)
- Siamese model shares one encoder across both inputs, outputs Euclidean distance via a custom `EuclideanDistance` layer
- Contrastive loss with margin = 1.0

### 4. Training (`train.py`)
- Multi-GPU via `tf.distribute.MirroredStrategy`
- **Progressive unfreeze schedule** (current version):
  - Epochs 0–4: backbone fully frozen, only the head trains (6 trainable variables)
  - Epochs 5, 8, 11, 14: unlock 5 more backbone layers each stage (deepest layers first, BatchNorm layers excluded from unfreezing throughout), LR stepped down at each stage (`1e-5 → 5e-6 → 2e-6 → 1e-6`)
  - 20 layers unfrozen in total by epoch 14, with epochs 14–19 to train on the fully-staged trainable set
- Optimizer must be **recreated inside `strategy.scope()`** every time the trainable-variable set changes (each unfreeze stage) — Keras 3 optimizers lock to the variable set they were first built with, and silently error otherwise
- Checkpointing: best model saved on validation-loss improvement only (`best_model.weights.h5`)
- LR halved after 3 consecutive epochs with no val-loss improvement (this is plateau-based LR decay, **not early stopping** — training always runs the full epoch count regardless of val performance; a real stopping condition is still on the to-do list)

### 5. Known data-pipeline pitfalls (for anyone touching this code)
- `tf.data.Dataset.cache()` should **never** be applied to the training dataset here — pairs are regenerated every epoch (by design, for hard-negative mining), so nothing is ever reused, and caching decoded images in RAM caused repeated OOM kernel restarts (~13GB per cached epoch)
- `.shuffle()` should be applied to raw file paths/labels, before `.map(load_and_resize)` — shuffling after decode buffers full decoded images instead of cheap strings, at ~1.2MB per pair
- CUB-200-2011 contains a small number of corrupt/non-standard JPEG files — validate images once after loading the split, before generating any pairs

---

## Person B — Embedding Index, Retrieval, Evaluation, Visualization

*(Scope carried out by the same person as Person A for this iteration of the project — kept as a separate section since the responsibilities are distinct.)*

### 1. Fixed evaluation set
Before computing any final metrics, generate a **larger, fixed** set of val/test pairs once and reuse it — do not evaluate off the training loop's per-epoch (randomly resampled) val loss. This is what any reported precision@k or final val-loss figure should be computed against.

### 2. Embedding index (`embed_index.py`)
Load `best_model.weights.h5`, run every image belonging to `test` (and optionally `val`) species through the encoder, and cache `{image_path: embedding}` to disk. This is exhaustive, unlike the 8-per-class sample used for training-time hard-negative mining.

### 3. Retrieval + evaluation (`retrieval.py`, `evaluate.py`)
k-NN search over the embedding index (`sklearn.neighbors.NearestNeighbors` is sufficient at this dataset scale); precision@k computed against ground-truth species labels from `cub200_splits.json`.

### 4. Visualization (`visualize.py`)
t-SNE/UMAP projection of the embedding space, colored by species; similarity grids showing top-k retrieved neighbors for sample queries.

### 5. Streamlit demo (`app.py`)
Upload or select a query image → embed live with the exact same preprocessing pipeline used at training time (resize to 224×224, normalize to [0,1]) → display top-k nearest neighbors from the index with distances. Preprocessing mismatch between training and this app is the most likely silent failure point — verify against the spec above before treating results as meaningful.
