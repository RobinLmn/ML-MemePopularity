import tensorflow as tf
import numpy as np
from tensorflow import keras
import processdata as data

def shuffle_in_unison(a, b):
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b

def process():
    size = 3225
    images, labels = data.get(size)

    print("Processing...")

    images, labels = shuffle_in_unison(images, labels)

    split = int(size * 0.9)
    return (images[0:split], labels[0:split]), (images[split:], labels[split:]),

def fit():

    (x_train, y_train), (x_test, y_test) = process()

    model = tf.keras.Sequential([
      tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dense(4)
    ])


    model.compile(
      optimizer='adam',
      loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=['accuracy'])

    print("Training...")
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=20)
    model.save("Images CNN : Thu Mar 18 2021")

def get():
    return tf.keras.models.load_model("../data/model_3")


if __name__ == '__main__':
    fit()
