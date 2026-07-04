import tensorflow as tf

class L2Normalize(tf.keras.layers.Layer):
    def call(self, x):
        return tf.math.l2_normalize(x, axis=1)

class EuclideanDistance(tf.keras.layers.Layer):
    def call(self, tensors):
        return tf.norm(tensors[0] - tensors[1], axis=1)
    
layers = tf.keras.layers
Model = tf.keras.Model
regularizers = tf.keras.regularizers

def build_encoder(input_shape=(224, 224, 3), embedding_dim=128, freeze_backbone=True, l2_reg=1e-4):
    inputs = layers.Input(shape=input_shape, name="image")
    base_model = tf.keras.applications.ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape,
        pooling='avg'
    )
    base_model.trainable = not freeze_backbone
    x = base_model(inputs)
    x = layers.Dense(512, activation="relu", kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(embedding_dim, kernel_regularizer=regularizers.l2(l2_reg))(x)
    embeddings = L2Normalize()(x)
    return Model(inputs, embeddings, name='encoder')

def build_siamese_model(enc):
    img1 = layers.Input(shape=(224, 224, 3), name='img1')
    img2 = layers.Input(shape=(224, 224, 3), name='img2')
    tensor1 = enc(img1)
    tensor2 = enc(img2)
    distance = EuclideanDistance()([tensor1, tensor2])
    return Model([img1, img2], distance)

if __name__ == "__main__":
    from pathlib import Path
    ROOT = Path(__file__).resolve().parent.parent
    enc = build_encoder()
    model = build_siamese_model(enc)
    model.load_weights(str(ROOT / "artifacts" / "best_model.weights.h5"))
    encoder = model.get_layer('encoder')
    print(f'model loaded')
    print(encoder.output_shape)

    for layer in model.layers:
        print(layer)