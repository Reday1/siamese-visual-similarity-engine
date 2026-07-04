import os
import json
from PIL import Image


DATA_ROOT = "data/CUB_200_2011"
IMAGE_DIR = os.path.join(DATA_ROOT, "images")

OLD_SPLIT = "artifacts/cub200_dataset_splits.json"
NEW_SPLIT = "artifacts/cub200_dataset_manifest.json"


def valid_image(path):
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False

def add_image_paths(species_list):

    records = []

    for species in species_list:

        species_path = os.path.join(
            IMAGE_DIR,
            species
        )

        for img_name in os.listdir(species_path):

            abs_path = os.path.join(
                species_path,
                img_name
            )

            if not valid_image(abs_path):
                print("Skipping corrupt:", abs_path)
                continue

            rel_path = os.path.join(
                "images",
                species,
                img_name
            )

            records.append(
                {
                    "path": rel_path,
                    "species": species
                }
            )

    return records


def build_manifest():

    # don't overwrite final manifest
    if os.path.exists(NEW_SPLIT):
        with open(NEW_SPLIT, "r") as f:
            return json.load(f)


    # load original training split
    with open(OLD_SPLIT, "r") as f:
        old = json.load(f)


    manifest = {

        "train": add_image_paths(
            old["train_species"]
        ),

        "val": add_image_paths(
            old["val_species"]
        ),

        "test": add_image_paths(
            old["test_species"]
        )
    }

    with open(NEW_SPLIT, "w") as f:
        json.dump(
            manifest,
            f,
            indent=4
        )

    return manifest

if __name__ == "__main__":

    manifest = build_manifest()

    print("Train images:", len(manifest["train"]))
    print("Val images:", len(manifest["val"]))
    print("Test images:", len(manifest["test"]))