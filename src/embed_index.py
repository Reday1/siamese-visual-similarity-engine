import numpy as np
import json
import tensorflow as tf
from pathlib import Path

from model import build_encoder, build_siamese_model
from preprocessing import load_resize

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"
DATA_ROOT = ROOT / "data" / "CUB_200_2011"

enc = build_encoder()
model = build_siamese_model(enc)
model.load_weights(str(ARTIFACTS / "best_model.weights.h5"))

enc = model.get_layer('encoder')

with open(ARTIFACTS / "cub200_dataset_manifest.json", "r") as f:
    data = json.load(f)

embeddings = []
image_index = []

for item in data["test"]:
    img_path = str(DATA_ROOT / item["path"])
    img = load_resize(img_path)
    img = tf.expand_dims(img, axis = 0)
    emb = enc.predict(img, verbose = 0)
    embeddings.append(emb[0])
    image_index.append(item)

np.save(
    str(ARTIFACTS / "embeddings.npy"),
    np.array(embeddings)
)

with open(ARTIFACTS / "image_index.json", "w") as f:
    json.dump(
        image_index,
        f,
        indent = 4
    )
print(np.array(embeddings).shape)