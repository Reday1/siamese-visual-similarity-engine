import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

from pairs import make_pairs
from model import build_encoder, build_siamese_model
from dataset import create_dataset, load_character_images

os.makedirs('outputs', exist_ok=True)
os.makedirs('checkpoints', exist_ok=True)

char_dict = load_character_images('data/images_background_small1/images_background_small1')
pairs = make_pairs(char_dict)
dataset = create_dataset(pairs)

enc = build_encoder(freeze_backbone=False)
siamese = build_siamese_model(enc)

# L = y*d² + (1-y)*max(margin-d, 0)²
# Hadsell, Chopra, LeCun — CVPR 2006
def contrastive_loss(labels, distances, margin=1.0):
    labels = tf.cast(labels, tf.float32)
    same = labels * tf.square(distances)
    diff = (1 - labels) * tf.square(tf.maximum(margin - distances, 0))
    return tf.reduce_mean(same + diff)

optimizer = tf.keras.optimizers.Adam(1e-5)
epochs = 10
epoch_losses = []
best_loss = float('inf')
patience = 3
no_improve = 0

for epoch in range(epochs):
    batch_losses = []
    for (img1, img2), labels in dataset:
        with tf.GradientTape() as tape:
            distances = siamese([img1, img2])
            loss = contrastive_loss(labels, distances, margin=1.0)
        grads = tape.gradient(loss, siamese.trainable_variables)
        optimizer.apply_gradients(zip(grads, siamese.trainable_variables))
        batch_losses.append(loss.numpy())

    epoch_loss = np.mean(batch_losses)
    epoch_losses.append(epoch_loss)
    print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")

    siamese.save(f'checkpoints/siamese_unfrozen_epoch{epoch+1}.keras')

    if epoch_loss < best_loss - 1e-5:
        best_loss = epoch_loss
        no_improve = 0
        enc.save('outputs/encoder_unfrozen_best.keras')
        print(f"  -> best encoder saved (loss={best_loss:.4f})")
    else:
        no_improve += 1
        print(f"  -> no improvement ({no_improve}/{patience})")
        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

enc.save('outputs/encoder_unfrozen_final.keras')
print("Final encoder saved to outputs/encoder_unfrozen_final.keras")

plt.plot(epoch_losses, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Curve (unfrozen backbone)')
plt.savefig('outputs/training_curve_unfrozen.png', dpi=150, bbox_inches='tight')
plt.close()
print("Loss curve saved to outputs/training_curve_unfrozen.png")