import os
import tensorflow as tf
def load_character_images(root_dir):
    character_to_images = {}
    for alphabet in os.listdir(root_dir):
        alphabet_path = os.path.join(root_dir, alphabet)
        if not os.path.isdir(alphabet_path):
            continue
        for character in os.listdir(alphabet_path):
            character_path = os.path.join(alphabet_path, character)
            if not os.path.isdir(character_path):
                continue
            full_paths = [os.path.join(character_path, f) for f in os.listdir(character_path)]
            key = alphabet + "_" + character
            character_to_images[key] = full_paths
    return character_to_images

def proc_img(path1, path2, label):
    img1 = tf.io.read_file(path1)
    img2 = tf.io.read_file(path2)

    img1 = tf.image.decode_png(img1, channels = 3)
    img2 = tf.image.decode_png(img2, channels =3)


    img1 = tf.image.resize(img1, (224,224))
    img2 = tf.image.resize(img2, (224,224))

    img1 = tf.cast(img1, tf.float32) / 255.0
    img2 = tf.cast(img2, tf.float32) / 255.0
    return (img1, img2), label

def create_dataset(pairs, batch_size=32, shuffle = True):
    #pairs is of form [[img1,img2, label], ... .]
    img1_paths = []
    img2_paths=[]
    labels = []
    for pair in pairs:
        img1_paths.append(pair[0])
        img2_paths.append(pair[1])
        labels.append(pair[2])
    dataset = tf.data.Dataset.from_tensor_slices(((img1_paths, img2_paths), labels))
    dataset = dataset.map(lambda paths, label: proc_img(paths[0], paths[1], label))
    dataset = dataset.shuffle(1000)
    dataset = dataset.batch(8) 
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset
