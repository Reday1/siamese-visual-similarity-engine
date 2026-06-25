import os

def load_character_images(root_dir):
    """
    root_dir: path to a folder containing alphabet folders.
    Example: data/images_background

    Returns a dictionary:
      key   = "AlphabetName_characterXX"
      value = list of full image paths for that character
    """
    character_to_images = {}

    for alphabet in os.listdir(root_dir):
        alphabet_path = os.path.join(root_dir, alphabet)

        for character in os.listdir(alphabet_path):
            character_path = os.path.join(alphabet_path, character)
            image_files = os.listdir(character_path)
            full_paths = [os.path.join(character_path, f) for f in image_files]

            key = alphabet + "_" + character
            character_to_images[key] = full_paths

    return character_to_images