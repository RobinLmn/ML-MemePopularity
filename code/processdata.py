import json
import numpy as np

import PIL.Image

import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt

import requests
from io import BytesIO




def url_to_image(url):
    response = requests.get(url)
    return PIL.Image.open(BytesIO(response.content))


def get_upvotes(data):
    upvotes = []

    for i in range(1, len(data)+1):
        if data[str(i)] != "Null":
          upvotes.append(data[str(i)]["ups"] - data[str(i)]["downs"])
    mean = np.mean(upvotes)

    for i in range(len(upvotes)):
        upvotes[i] = "funny" if upvotes[i] >= mean else "boring"
    return upvotes


def get_images(data):
    images = []
    for i in range(1, len(data)+1):
        try:
          img = url_to_image(data[str(i)]["media"])
          images.append(img)
        except:
          print(str(i) + " is sad :(")
          data[str(i)] = "Null"
    return images, data


def process_image(img):
    return tf.keras.preprocessing.image.img_to_array(img, data_format=None, dtype=None)


def divide(data):
    size = len(data)
    training_size = int(size * 0.8)
    test_size = int(size)
    return data[0:training_size], data[training_size: test_size]


def get():
    response = requests.get("https://raw.githubusercontent.com/RobinLmn/ML-MemePopularity/main/data/db.json")
    data = response.json()["_default"]

    images, data = get_images(data)
    upvotes = get_upvotes(data)

    assert(len(images) == len(upvotes))

    return divide(images), divide(upvotes)


def main():
    print("Loading data...")
    images, labels = get()
    train_images, test_images = images
    train_labels, test_labels = labels

    print("Processing images...")
    train_images = list(map(process_image, train_images))
    test_images = list(map(process_image, test_images))

    print("Finished processing")
    return train_images, test_images, train_labels, test_labels

if __name__ == '__main__':
    main()
