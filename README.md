# Siamese Visual Similarity Engine — README

## What this project is
A Siamese neural network trained to learn a general visual-similarity embedding space — given two images, output a distance that's small if they're the same class and large if they're different — rather than a fixed-class classifier. The encoder is trained via contrastive loss on image *pairs*, never on class labels directly, so the resulting similarity function is designed to generalize to categories it has never seen (zero-shot similarity).

Two datasets were used in sequence:
1. **Omniglot** — proof-of-concept phase, full pipeline built and validated here first.
2. **CUB-200-2011** — 200 fine-grained bird species, applied second once the pipeline was proven end to end. This is the dataset behind the final trained checkpoint.

**Zero-shot framing:** train/val/test splits are made by *species*, not by image — ~40 of 200 species are held out entirely and never appear during training. This tests whether the model's learned notion of "same vs. different" transfers to species it has no direct experience of, rather than testing memorization of 200 known classes.

---

## What each notebook file is

| File | Status | Purpose |
|---|---|---|
| `notebook0f3aae9405.ipynb` | ✅ **Source of truth** | The corrected, working training notebook. This is the exact code (fixed decode function, real custom layers, staged unfreeze schedule) that produced the final checkpoint, `best_model.weights.h5`, over a 20-epoch run. Use this as the reference for what the model actually is. |
| `training_step.ipynb` | ⚠️ **Deprecated / do not use as reference** | An earlier, partially-fixed branch that still contains the old buggy code (JPEG files decoded with `decode_png` instead of `decode_image`, `Lambda`-based distance/normalization layers instead of the named `EuclideanDistance`/`L2Normalize` classes, and literal placeholder `...` stubs in `build_encoder`/`build_siamese_model`). Kept for history only — running it as-is will crash or silently diverge from the trained checkpoint. |
| `cub200_dataset_splits.json` | ✅ Locked record | Species-name lists (`train_species`, `val_species`, `test_species`) defining the permanent train/val/test boundary. **Important caveat:** this notebook does not currently load from this file automatically — it was saved as a record after the fact, not wired into the split logic. See "Known gaps" below. |
| `best_model.weights.h5` | ✅ Final checkpoint | Trained Siamese model weights from the 20-epoch staged-unfreeze run on CUB-200. The ResNet50 backbone inside is **fine-tuned, not stock** — it will not behave like `ResNet50(weights='imagenet')` and must always be loaded via this repo's exact `build_encoder()`/`build_siamese_model()` definitions, not reconstructed from scratch. |
| Two-phase training PNGs | Reference only | Loss curves from the first (15-epoch, full-unfreeze) and second (20-epoch, staged-unfreeze) training runs. |

---

## Architecture summary
- **Backbone:** ResNet50, ImageNet-pretrained, `pooling='avg'`, no top.
- **Projection head:** `Dense(512, relu) → Dropout(0.3) → Dense(256, relu) → Dropout(0.3) → Dense(128)`.
- **Embedding:** L2-normalized via a custom `L2Normalize` layer (not `Lambda`, for clean serialization).
- **Siamese distance:** Euclidean distance between the two branch embeddings, via a custom `EuclideanDistance` layer.
- **Loss:** Contrastive loss, margin = 1.0.

## Training procedure (final run)
- Multi-GPU via `tf.distribute.MirroredStrategy`.
- **Progressive/staged backbone unfreeze**, not all-at-once:
  - Epochs 0–4: backbone fully frozen, only the head trains (6 trainable variables).
  - Epochs 5, 8, 11, 14: unlock 5 more backbone layers each stage (deepest layers first; BatchNorm layers excluded from unfreezing throughout to avoid instability at small batch sizes), with LR stepped down at each stage (`1e-5 → 5e-6 → 2e-6 → 1e-6`).
  - 20 layers unfrozen in total by epoch 14; epochs 14–19 train on the full staged trainable set.
- **Online hard-negative mining:** from epoch 3 onward, half of each epoch's negative pairs are "hard" — mined by embedding a sample of training images with the *current* encoder and selecting the closest cross-species match, so the negative pool gets harder as the model improves.
- **Checkpointing:** best model saved only on validation-loss improvement.
- **LR plateau reduction:** LR halved after 3 consecutive epochs with no val-loss improvement. This is **not early stopping** — the loop always runs the full epoch count regardless of val performance.

---

## Bugs found and fixed during development
These were all real, encountered issues — documented here so they aren't rediscovered later:

1. **`load_and_resize` decode bug:** an earlier version decoded `img1` twice instead of decoding `img2` from its own bytes — caused a crash and, in a subtler form, would have silently paired every image with itself.
2. **Wrong decoder for file format:** a later branch reverted to `tf.image.decode_png` on CUB-200's actual JPEG files — replaced with `tf.image.decode_image(..., expand_animations=False)`, which handles both formats.
3. **Keras 3 optimizer state bug:** Keras 3 optimizers lock to the specific variable set they were first built with. Every time the trainable-variable set changes (e.g. at an unfreeze stage), the optimizer must be recreated fresh inside `with strategy.scope():` — reusing or reassigning LR on the old optimizer object throws `Unknown variable` errors.
4. **OOM / repeated kernel restarts:** caused by two compounding issues in `create_dataset` — `.cache()` was caching ~13GB of fully-decoded images in RAM for a dataset that's regenerated every epoch and never reused (hard-negative mining means train pairs differ every epoch), and `.shuffle()` was placed *after* decoding, buffering full images instead of cheap file-path strings. Fixed by removing `.cache()` entirely and moving `.shuffle()` before the decode `.map()` call.
5. **Corrupt/non-standard JPEGs in CUB-200-2011:** a small number of dataset files fail TensorFlow's JPEG decoder (`Invalid JPEG data or crop window`). Requires either pre-validating images with `PIL.Image.verify()` after loading the split, and/or wrapping the dataset with `tf.data.experimental.ignore_errors()` as a safety net.
6. **`No such layer: encoder` error:** caused by stale function definitions — an old, `Lambda`-based `build_encoder`/`build_siamese_model` (no `name='encoder'`) got executed *after* the corrected versions due to out-of-order cell execution, so `enc`/`siamese` were built from the wrong definitions. Root-caused to notebook branches forking from different points in development.
7. **Species split not locked across sessions:** `load_and_split_cub_images` uses `train_test_split(..., random_state=42)`, which is reproducible *given* a fixed input order — but `os.listdir()` order isn't guaranteed stable across fresh dataset re-extractions, so the actual species assigned to train/val/test could shift between sessions even with the same seed. This is the most serious of the bugs found, since it silently breaks the zero-shot evaluation's validity (a species meant to be permanently held out could end up in training in a later session) without throwing any error.

---

## Known gaps (not yet fixed)
- **Split-locking is not wired into the current notebook.** `cub200_dataset_splits.json` exists as a saved record, but `load_and_split_cub_images` is still called fresh every run with no check for an existing split file. This should be patched before ever re-running this notebook for a new training pass.
- **No real early stopping** — only LR plateau reduction exists; the training loop always completes all epochs regardless of validation trend.
- **Validation pairs are regenerated every epoch** rather than held fixed, which adds epoch-to-epoch noise to the val loss curve. Final reported metrics should come from a separately-generated, fixed, larger evaluation set — not from the training loop's per-epoch val loss.
- **Train vs. val loss are not directly comparable in the final curve** — training pairs include hard negatives (mined, confusable pairs) from epoch 3 onward, while validation pairs are always random. Val loss appearing lower than train loss is partly a reflection of this difficulty asymmetry, not purely a generalization signal.

---

## Current status
- [x] Omniglot proof-of-concept trained and validated.
- [x] CUB-200 species-disjoint pipeline built (pairing, augmentation, contrastive loss, distributed training).
- [x] All bugs above identified and fixed in `notebook0f3aae9405.ipynb`.
- [x] Final 20-epoch staged-unfreeze training run completed; `best_model.weights.h5` saved.
- [x] Project moved from Kaggle to local machine (Fedora, RTX 4050) for the retrieval/eval/app phase.
- [ ] Split-locking patch not yet applied to the training notebook (low priority now that training is complete, but should be done before any retraining).
- [ ] Fixed/larger evaluation pair set for trustworthy final metrics — not yet built.
- [ ] Exhaustive embedding index over test (and val) species — not yet built.
- [ ] k-NN retrieval + precision@k evaluation — not yet built.
- [ ] t-SNE/UMAP visualization + similarity grids — not yet built.
- [ ] Streamlit app + backing API — in progress, next notebook.

---

## Next steps (in order)
1. Build a fixed, larger val/test pair set for trustworthy evaluation metrics.
2. Generate a full embedding index: run `best_model.weights.h5`'s encoder exhaustively over every test-species image (not the 8-per-class sample used for training-time hard-negative mining).
3. Build k-NN retrieval and compute precision@k against the fixed eval set.
4. t-SNE/UMAP visualization of the embedding space, plus similarity grids for sample queries.
5. Build an API layer around the trained encoder, and a Streamlit front end that calls it for live image upload → nearest-neighbor retrieval.

**Critical constraint for step 5:** whatever preprocessing pipeline the API uses (resize to 224×224, normalize to [0,1], same decode path) must exactly match `load_and_resize` from training. Any mismatch here degrades retrieval quality silently, with no error thrown.
