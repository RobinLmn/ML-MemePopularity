import json
import numpy as np
import PIL.Image
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt


def get_upvotes(data):
    upvotes = []

    for i in range(1, len(data)+1):
        upvotes.append(data[str(i)]["ups"] - data[str(i)]["downs"])
    mean = np.mean(upvotes)

    for i in range(len(upvotes)):
        upvotes[i] = "funny" if upvotes[i] >= mean else "boring"
    return upvotes


def get_images(data):
    images = []
    for i in range(1, len(data)+1):
        try:
            img = PIL.Image.open("../data/memes/" + data[str(i)]["id"] + ".png")
            images.append(img)
        except:
            try:
                img = PIL.Image.open("../data/memes/" + data[str(i)]["id"] + ".jpg")
                images.append(img)
            except:
                data.pop(str(i))
    return images

def process_image(img):
    return tf.keras.preprocessing.image.img_to_array(img, data_format=None, dtype=None)

def divide(data):
    size = len(data)
    training_size = int(size * 0.8)
    test_size = int(size)
    return data[0:training_size], data[training_size: test_size]


def get():
    with open("../data/db.json", "r") as db:
        data = json.load(db)["_default"]

        images = get_images(data)
        upvotes = get_upvotes(data)

        assert(len(images) == len(upvotes))

        return divide(images), divide(upvotes)

images, labels = get()
train_images, test_images = images
train_labels, test_labels = labels
