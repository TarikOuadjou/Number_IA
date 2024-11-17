import tensorflow as tf
import pandas as pd
import numpy as np

mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((train_images.shape[0], 28 * 28))
df = pd.DataFrame(train_images)
df.insert(0, 'label', train_labels)
df.to_csv('mnist_train.csv', index=False, header=False)

print("Fichier 'mnist_train.csv' créé avec succès!")
