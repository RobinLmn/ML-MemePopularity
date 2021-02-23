import tensorflow as tf
import numpy as np
from tensorflow import keras
import processdata as data




# Normalize the data, tensorflow works better with data between 0 and 1
training_images  = training_images / 255.0
test_images = test_images / 255.0

# Model
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),                              # First layer to flatten the data to a 1D array
                                    tf.keras.layers.Dense(128, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

# Compiler
model.compile(optimizer = tf.keras.optimizers.Adam(),
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Fit
model.fit(training_images, training_labels, epochs=5)
