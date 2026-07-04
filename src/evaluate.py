import json
import os
from pathlib import Path
from retrieval import retrieve

# we are going to asses when we give an input of a bird does our model give similar results or not 
# top k are 5 we expect the precision value as a ration of int / 5 whether similar birds came out of 5
#assesing on test split, train never saw these species

ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = str(ROOT / "data" / "CUB_200_2011")

INDEX_PATH = ROOT / "artifacts" / "image_index.json"


with open(INDEX_PATH, "r") as f:
    image_index = json.load(f)



def precision_at_k(k):
    total_score = 0
    total_images = len(image_index)
    
    for query_item in image_index:
        query_path = os.path.join(
            DATA_ROOT,
            query_item["path"]
        )
        true_species = query_item["species"]

        # k+1 because first result is itself
        results = retrieve(
            query_path,
            k + 1
        )

        # remove same image
        results = [
            r for r in results
            if r["path"] != query_item["path"]
        ]

        results = results[:k]


        correct = 0

        for r in results:
            if r["species"] == true_species:
                correct += 1


        total_score += correct / k



    return total_score / total_images



if __name__ == "__main__":

    p1 = precision_at_k(1)

    p5 = precision_at_k(5)


    print(
        "Precision@1:",
        p1
    )

    print(
        "Precision@5:",
        p5
    )