import os
import random
import json
import matplotlib.pyplot as plt
from PIL import Image

from retrieval import retrieve


DATA_ROOT = "/home/h/Documents/GitHub/siamese-visual-similarity-engine/data/CUB_200_2011"

INDEX_PATH = "/home/h/Documents/GitHub/siamese-visual-similarity-engine/artifacts/image_index.json"


with open(INDEX_PATH, "r") as f:
    image_index = json.load(f)


def show_retrieval(query_item, k=5):

    query_path = os.path.join(
        DATA_ROOT,
        query_item["path"]
    )

    results = retrieve(
        query_path,
        k + 1
    )

    # remove itself
    results = [
        r for r in results
        if r["path"] != query_item["path"]
    ][:k]


    plt.figure(figsize=(15, 4))


    # query image
    plt.subplot(1, k + 1, 1)

    img = Image.open(query_path)

    plt.imshow(img)
    plt.axis("off")

    plt.title(
        "QUERY\n" +
        query_item["species"],
        fontsize=8
    )


    # retrieved images
    for i, item in enumerate(results):

        img_path = os.path.join(
            DATA_ROOT,
            item["path"]
        )

        img = Image.open(img_path)

        plt.subplot(
            1,
            k + 1,
            i + 2
        )

        plt.imshow(img)
        plt.axis("off")


        match = (
            "✓"
            if item["species"] == query_item["species"]
            else "X"
        )

        plt.title(
            match
            + "\n"
            + item["species"]
            + "\n"
            + f'{item["distance"]:.2f}',
            fontsize=7
        )


    plt.tight_layout()
    plt.show()



if __name__ == "__main__":

    print("started")

    sample = random.choice(image_index)

    print("query:", sample["species"])

    show_retrieval(
        sample,
        k=5
    )

    print("done")