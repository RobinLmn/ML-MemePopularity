import tensorflow as tf
import numpy as np
from tensorflow import keras
import processdata as data

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, BatchNormalization, Conv2D, Dense, MaxPooling2D, Flatten
# Regularization tools
from tensorflow.keras.layers import Dropout
from tensorflow.keras import regularizers

import matplotlib.pyplot as plt

from tensorflow.keras.callbacks import LearningRateScheduler

import h5py

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

    h5f = h5py.File('data.h5', 'w')
    h5f.create_dataset('x_train', data=images[0:split])
    h5f.create_dataset('y_train', data=labels[0:split])
    h5f.create_dataset('x_test', data=images[split:])
    h5f.create_dataset('y_test', data=labels[split:])
    h5f.close()

def fit():

    h5f = h5py.File('data.h5','r')
    x_train = h5f['x_train'][:]
    y_train = h5f['y_train'][:]
    x_test = h5f['x_test'][:]
    y_test = h5f['y_test'][:]
    h5f.close()

    weight_decay = 1e-4
    # Instantiate model
    model = Sequential()

    # Add first convolutional layer
    model.add(Conv2D(32, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), input_shape=x_train.shape[1:]))
    model.add(Activation('elu'))
    model.add(BatchNormalization())

    # Add second convolutional layer
    model.add(Conv2D(32, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))

    # Add third convolutional layer
    model.add(Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.3))

    # Add fourth convolutional layer
    model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())

    # Add fifth and final convolutional layer
    model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.4))

    # Flatten and add last layer to output predictions
    model.add(Flatten())
    model.add(Dense(4, activation='softmax'))


    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if(logs.get('accuracy')>0.95):
                print("\nReached 95% accuracy so cancelling training!")
                self.model.stop_training = True

    # Change the learning rate depending on the current epoch
    def lr_schedule(epoch):
        lrate = 0.001
        if epoch > 75:
            lrate = 0.0005
        if epoch > 100:
            lrate = 0.0003
        return lrate


    callbacks = myCallback()
    opt_rms = keras.optimizers.RMSprop(lr=0.001,decay=1e-6)

    print("Training...")
    model.compile(loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer=opt_rms, metrics=['accuracy'])
    model = model.fit(x_train, y_train, epochs=150,
                        verbose=1, validation_data=(x_test,y_test),
                        callbacks=[LearningRateScheduler(lr_schedule), callbacks])

    model.save("model")

def evaluate():
    history = get()
    print("Evaluating...")
    print(history.history.keys())
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Augmented and Regularized CNN')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='lower right')
    plt.show()


def get():
    return tf.keras.models.load_model("../data/model_3")


if __name__ == '__main__':
    fit()
    evaluate()
