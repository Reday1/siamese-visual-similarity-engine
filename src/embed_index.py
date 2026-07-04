import numpy as np
import json
import tensorflow as tf

from model import build_encoder, build_siamese_model
from preprocessing import load_resize

enc = build_encoder()
model = build_siamese_model(enc)
model.load_weights("/home/h/Documents/GitHub/siamese-visual-similarity-engine/artifacts/best_model.weights.h5")

enc = model.get_layer('encoder')

pth = "/home/h/Documents/GitHub/siamese-visual-similarity-engine/artifacts/cub200_dataset_manifest.json"

with open(pth, "r") as f:
    data = json.load(f)

embeddings = []
image_index = []

data_root = "/home/h/Documents/GitHub/siamese-visual-similarity-engine/data/CUB_200_2011"

for item in data["test"]:
    img_path = data_root + "/" + item["path"]
    img = load_resize(img_path)
    img = tf.expand_dims(img, axis = 0)
    emb = enc.predict(img, verbose = 0)
    embeddings.append(emb[0])
    image_index.append(item)

np.save(
    "/home/h/Documents/GitHub/siamese-visual-similarity-engine/artifacts/embeddings.npy",
    np.array(embeddings)
)

with open("/home/h/Documents/GitHub/siamese-visual-similarity-engine/artifacts/image_index.json", "w") as f:
    json.dump(
        image_index,
        f,
        indent = 4
    )
print(np.array(embeddings).shape)