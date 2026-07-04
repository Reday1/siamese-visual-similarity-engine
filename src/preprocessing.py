import tensorflow as tf
def load_resize(imgPath : str):
    img = tf.io.read_file(imgPath)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, (224, 224))
    img = tf.cast(img, tf.float32) / 255.0
    return img 
