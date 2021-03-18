import tensorflow as tf
import numpy as np
from tensorflow import keras
import processdata as data

def fit():

    size = 3225
    images, labels = data.get(size)

    print("Processing...")

    dataset = tf.data.Dataset.from_tensor_slices((np.array(images), np.array(labels)))
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(32)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    train, test = dataset.take(int(size * 0.8)), dataset.skip(int(size * 0.8))

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
    model.fit(train, validation_data=test, epochs=5)
    model.save("Images CNN : Tue Mar 16 2021 (2)")

def get():
    return tf.keras.models.load_model("../data/model")

if __name__ == '__main__':
    fit()
