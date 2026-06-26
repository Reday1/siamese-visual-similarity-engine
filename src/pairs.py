import os
import dataset
import random


# what are doing exactly?
# we are trying to create labelled data, of form [img1, img2, 0 or 1]
#
#  0 means different 1 means same
# as input we are using load_char function from dataset.py
# this gives us a dictionary of {image name: location} is basically like a vector table
#
# iteration 1 : dataset : omniglot :964 classes 20 image per class
# taking 20 same 20 different per class, 20 images all different for same labels

def make_pairs(char_dict, num_pairs=20):
    class_names = list(char_dict.keys())
    all_pairs = []

    for current_class in class_names:
        images = char_dict[current_class]

        # ---- SAME PAIRS (label = 1) ----
        used_same = set()
        while len(used_same) < num_pairs:
            img1 = random.choice(images)
            img2 = random.choice(images)
            while img2 == img1:
                img2 = random.choice(images)

            pair_check = tuple(sorted([img1, img2]))
            if pair_check in used_same:
                continue

            used_same.add(pair_check)
            all_pairs.append([img1, img2, 1])

        # ---- DIFFERENT PAIRS (label = 0) ----
        used_diff = set()
        while len(used_diff) < num_pairs:
            class2 = random.choice([x for x in class_names if x != current_class])
            img1 = random.choice(images)
            img2 = random.choice(char_dict[class2])

            pair_check = tuple(sorted([img1, img2]))
            if pair_check in used_diff:
                continue

            used_diff.add(pair_check)
            all_pairs.append([img1, img2, 0])

    random.shuffle(all_pairs)
    return all_pairs
#--sanity check--
# char_dict = dataset.load_character_images('data/images_background_small1/images_background_small1')
# pairs = make_pairs(char_dict, num_pairs=20)
# print(len(pairs))
# print(pairs[0])