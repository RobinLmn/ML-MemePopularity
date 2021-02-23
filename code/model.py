import tensorflow as tf
import numpy as np
from tensorflow import keras
import processdata as data

def fit():

    training_images, test_images, train_labels, test_labels = data.main()

    training_images  = training_images / 255.0
    test_images = test_images / 255.0

    model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),                          
                                        tf.keras.layers.Dense(128, activation=tf.nn.relu),
                                        tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

    model.compile(optimizer = tf.keras.optimizers.Adam(),
                  loss = 'sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    print("Training...")
    model.fit(training_images, training_labels, epochs=5)

if __name__ == '__main__':
    fit()
