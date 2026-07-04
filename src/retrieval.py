import numpy as np
import tensorflow as tf
from model import *
import preprocessing
import json

enc = build_encoder()
model = build_siamese_model(enc)
model.load_weights('/home/h/Documents/GitHub/siamese-visual-similarity-engine/artifacts/best_model.weights.h5')
encoder = model.get_layer('encoder')

database_embeddings = np.load("/home/h/Documents/GitHub/siamese-visual-similarity-engine/artifacts/embeddings.npy")

with open("/home/h/Documents/GitHub/siamese-visual-similarity-engine/artifacts/image_index.json", "r") as f:
    ind = json.load(f)
def retrieve(query_path, k):
    query = preprocessing.load_resize(query_path)
    query = tf.expand_dims(
        query,
        axis=0
    )
    emb = encoder.predict(
        query,
        verbose=0
    )[0]

    dist = np.linalg.norm(
        database_embeddings - emb,
        axis=1
    )
    top_k = np.argsort(dist)[:k]
    results = []
    for idx in top_k:
        item = ind[idx].copy()
        item["distance"] = float(dist[idx])
        results.append(item)
    return results