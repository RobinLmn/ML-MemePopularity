import tensorflow as tf
import numpy as np
from tensorflow import keras
import processdata as data

def fit():

    size = 3225
    training_images, test_images, training_labels, test_labels = data.get(size)

    print("Processing...")

    train_dataset = tf.data.Dataset.from_tensor_slices((np.array(training_images), np.array(training_labels)))
    train_dataset = train_dataset.cache()
    train_dataset = train_dataset.shuffle(buffer_size=1000)
    train_dataset = train_dataset.batch(32)
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    test_dataset = tf.data.Dataset.from_tensor_slices((np.array(test_images), np.array(test_labels)))
    test_dataset = test_dataset.cache()
    test_dataset = test_dataset.shuffle(buffer_size=1000)
    test_dataset = test_dataset.batch(32)
    test_dataset = test_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    model = tf.keras.Sequential([
      tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dense(2)
    ])


    model.compile(
      optimizer='adam',
      loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=['accuracy'])

    print("Training...")
    model.fit(train_dataset, validation_data=test_dataset, epochs=3)

if __name__ == '__main__':
    fit()
