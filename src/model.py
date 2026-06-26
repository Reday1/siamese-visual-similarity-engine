import tensorflow as tf
layers = tf.keras.layers
Model = tf.keras.Model
import numpy as np
# take input of one image 224 224 3 
# train this on tf.keras.applications.ResNet50 with last layer removed to output a flat vector
# resnet is trained on imagenet, thus we can just freeze the weights,  and parameter named freeze backbone is added
# small dataset == freeze the resnet parameters. unfreeze for larger datasets
# gives a 2048 dim vector 
# pass this vector through dense layers to reduce it to 128 bits
# use l2 normalization on 128 dim vector

# build siamese wrapper that calls encoder twice for 2 images
# computes distance between 2 resultant embeddings
# outputs this distance
# distance compared with label during training

def build_encoder(input_shape=(224, 224, 3), embedding_dim = 128, freeze_backbone = False):
        inputs = layers.Input(shape=input_shape, name="image")
        
        base_model = tf.keras.applications.ResNet50(
        weights ='imagenet',
        include_top = False,
        input_shape = input_shape,
        pooling = 'avg'
        )
        base_model.trainable = not freeze_backbone


        x = base_model(inputs)
        x = layers.Dense(512, activation="relu")(x)
        x = layers.Dense(256, activation="relu")(x)
        x = layers.Dense(embedding_dim)(x)
        embeddings = layers.Lambda(lambda t : tf.math.l2_normalize(t, axis = 1))(x)
        encoder = Model(inputs, embeddings)
        return encoder


# build siamese model takes input of images and gives output of distance
def build_siamese_model(enc):
        img1 = layers.Input(shape = (224, 224, 3), name = 'img1')
        img2 = layers.Input(shape = (224, 224, 3), name = 'img2')

        tensor1 = enc(img1)
        tensor2 = enc(img2)

        #creating model like structure, making a layer to calculate distance, 
        # had to use tf.norm to find the pythagorean type distance for 128 dimensions 
        distance = layers.Lambda(lambda tensors :tf.norm(tensors[0]-tensors[1],axis=1))([tensor1, tensor2])   

        siamese_model = Model([img1, img2], distance)  
        return siamese_model

#--sanity check--
# enc = build_encoder(freeze_backbone=True)
# siamese = build_siamese_model(enc)

# dummy_img1 = np.random.rand(2,224,224,3).astype('float32')
# dummy_img2 = np.random.rand(2,224,224,3).astype('float32')

# distance = siamese([dummy_img1, dummy_img2])

# print(f"output shape {distance.shape}")
# print(f"calculated distance:{distance.numpy()}")