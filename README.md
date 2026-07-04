# Siamese Visual Similarity Engine

A zero-shot visual similarity retrieval system using a Siamese Neural Network trained with contrastive learning.

Instead of classifying an image into fixed categories, the model learns an embedding space where visually similar images are close together. This allows retrieval of similar images from species that were never seen during training.

---

## Overview

Traditional classifiers answer:

```
What class is this image?
```

This project answers:

```
Which images are visually most similar?
```

The system:

1. Encodes an input image into a learned feature vector.
2. Compares it against a database of stored embeddings.
3. Retrieves nearest neighbors based on embedding distance.

This is similar in principle to systems used for:

- image search
- face verification
- product retrieval
- visual recommendation systems

---

## Dataset

Dataset:

**CUB-200-2011 Birds Dataset**

- 200 fine-grained bird species
- ~11k images

The split is performed by species, not images.

Example:

```
Train:
species A, B, C

Test:
species X, Y, Z
```

Test species are never shown during training.

This evaluates whether the network learned a general similarity function instead of memorizing classes.

---

## Model Architecture

### Backbone

```
ResNet50
(ImageNet pretrained)
↓
Global Average Pooling
```

The backbone is fine-tuned during training.

### Embedding Head

```
Dense(512)
ReLU
Dropout(0.3)

Dense(256)
ReLU
Dropout(0.3)

Dense(128)

L2 Normalization
```

Final output:

```
128-dimensional embedding vector
```

---

## Siamese Network

Two images are passed through the same encoder:

```
Image A ──┐
          │
          ├── Shared Encoder ── Embedding A
          │

Image B ──┘
                 |
                 v

        Euclidean Distance

                 |
                 v

          Contrastive Loss
```

Loss:

```
Contrastive Loss
Margin = 1.0
```

The model learns:

- pull similar images closer
- push different images apart

---

## Training Details

Training was performed with progressive fine tuning.

### Stage 1

Frozen ResNet50:

```
Train embedding head only
```

### Stage 2

Gradually unfreeze deeper ResNet layers:

```
Epoch 5  → +5 layers
Epoch 8  → +5 layers
Epoch 11 → +5 layers
Epoch 14 → +5 layers
```

Batch Normalization layers remained frozen.

Learning rates:

```
1e-5
5e-6
2e-6
1e-6
```

Additional techniques:

- online hard negative mining
- checkpointing on validation loss
- progressive backbone unfreezing

---

## Retrieval Pipeline

After training:

```
Images
  |
  v

Encoder

  |
  v

128-D embeddings

  |
  v

embeddings.npy
```

During inference:

```
Query Image

      |

Encoder

      |

Query Embedding

      |

Euclidean Distance Search

      |

Top-K Similar Images
```

The model does not retrain when new images are added.

---

## Project Structure

```
src/

model.py
    Network architecture

preprocessing.py
    Image loading pipeline

build_splits.py
    Dataset manifest generation

embed_index.py
    Builds embedding database

retrieval.py
    Nearest neighbor search

evaluate.py
    Retrieval metrics

visualize.py
    Retrieval visualization


app/

streamlit_app.py
    Web interface
```

---

## Evaluation

Evaluation is performed only on unseen test species.

The model searches through test embeddings and retrieves nearest neighbors.

Results:

```
Precision@1: 34.6%

Precision@5: 28.9%
```

The task is zero-shot fine-grained retrieval, not classification.

Random retrieval among ~40 unseen species would be approximately:

```
~2.5%
```

---

## Example Retrieval

Input:

```
Bird image
```

Output:

```
Top visually similar birds:

1. species A
2. species A
3. visually similar species
4. visually similar species
5. visually similar species
```

Even incorrect species predictions often correspond to visually similar birds.

---

## Technologies

- Python
- TensorFlow / Keras
- ResNet50
- NumPy
- PIL
- Streamlit
- Matplotlib

---

## Key Features

- Zero-shot retrieval
- Siamese neural network
- Contrastive metric learning
- Fine-tuned CNN backbone
- Hard-negative mining
- Embedding database search
- Interactive image retrieval demo

---

## Future Improvements

- Triplet loss / supervised contrastive learning
- FAISS vector search
- Larger embedding gallery
- Vision Transformer backbone
- Web deployment

---

## Purpose

This project demonstrates learning a reusable visual similarity function rather than a fixed image classifier.

The trained encoder can compare new images without requiring retraining for new categories.
