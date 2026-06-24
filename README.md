# Siamese Visual Similarity Engine

A visual similarity retrieval system built with Siamese Networks. Instead of training a network to classify images into fixed categories, this project trains it to learn a *similarity function* — given two images, it learns whether they are visually alike or not, and produces embeddings that can be compared directly via distance.

## Core Idea

A single CNN backbone (shared weights) is used twice — once for each image in a pair. The two resulting embeddings are compared using a distance metric. The network is trained so that:

- Similar images (positive pairs) end up close together in embedding space
- Dissimilar images (negative pairs) end up far apart, up to a margin

Once trained, any new image can be embedded and compared against a stored index of embeddings to retrieve its nearest visual neighbors.

![Architecture Diagram](ADD_LINK_HERE)

## Tech Stack

- PyTorch
- OpenCV
- ResNet18 (backbone, pretrained)
- FAISS (or NumPy, for smaller datasets) for the embedding index
- scikit-learn / UMAP for embedding space visualization

## Pipeline

1. **Data preparation** — Organize a dataset with natural similarity groups (e.g., product categories, face identities, fashion items). Generate positive pairs (same class) and negative pairs (different class).
2. **Siamese model** — A shared ResNet18 encoder plus a small projection head, called twice per pair.
3. **Contrastive loss training** — Minimizes embedding distance for positive pairs, maximizes it (up to a margin) for negative pairs.
4. **Embedding index** — Every image in the dataset is passed through the trained encoder and stored as a vector.
5. **Retrieval** — A query image is embedded, then compared against the index to return the top-k nearest matches.
6. **Evaluation** — Precision@k, embedding space visualization, and qualitative similarity grids.

## Repository Structure
siamese-visual-similarity-engine/

├── data/

│   └── splits.json

├── src/

│   ├── dataset.py

│   ├── pairs.py

│   ├── model.py

│   ├── train.py

│   ├── embed_index.py

│   ├── retrieval.py

│   ├── evaluate.py

│   └── visualize.py

├── checkpoints/

│   └── encoder.pt

├── outputs/

│   ├── embeddings.npy

│   ├── labels.npy

│   └── filepaths.json

├── notebooks/

│   └── demo.ipynb

├── requirements.txt

└── README.md

## Setup

```bash
git clone https://github.com/<your-username>/siamese-visual-similarity-engine.git
cd siamese-visual-similarity-engine
pip install -r requirements.txt
```

## Training

```bash
python src/train.py --data_dir data/ --epochs 30 --margin 1.0 --batch_size 64
```

This trains the Siamese encoder using contrastive loss and saves the resulting weights to `checkpoints/encoder.pt`.

![Training Loss Curve](ADD_LINK_HERE)

![Positive vs Negative Distance Distribution](ADD_LINK_HERE)

## Building the Embedding Index

```bash
python src/embed_index.py --checkpoint checkpoints/encoder.pt --data_dir data/ --out outputs/
```

This embeds every image in the dataset and stores the resulting vectors, labels, and filepaths for later retrieval.

## Running a Similarity Search

```bash
python src/retrieval.py --query path/to/query_image.jpg --k 5
```

Returns the top-5 most visually similar images from the indexed dataset, along with their distances.

![Similarity Search Result Grid](ADD_LINK_HERE)

## Evaluation

```bash
python src/evaluate.py --embeddings outputs/embeddings.npy --labels outputs/labels.npy
```

Reports Precision@k across the test set, broken down by class.

| Metric | Score |
|---|---|
| Precision@1 | ADD_VALUE |
| Precision@5 | ADD_VALUE |
| Precision@10 | ADD_VALUE |

## Embedding Space Visualization

```bash
python src/visualize.py --embeddings outputs/embeddings.npy --labels outputs/labels.npy
```

Projects the learned embedding space into two dimensions to check whether visually similar classes naturally cluster together, despite the model never being trained on class labels directly.

![t-SNE / UMAP Embedding Visualization](ADD_LINK_HERE)

## Results Summary

A short qualitative summary of what worked well, what classes retrieved poorly, and any notes on hard-negative mining or margin tuning go here.

## Team

| Role | Responsibility |
|---|---|
| Person A | Data pairing pipeline, Siamese model architecture, contrastive loss training |
| Person B | Embedding index, retrieval logic, evaluation metrics, visualizations |

## License

MIT
