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


def get_data(data, max):

    print("Getting images...")
    images = []
    for i in range(1, max+1):
        percent = round(i * 100 / max, 2)
        print(str(percent) + "%", end="\r")
        try:
            img = PIL.Image.open("../data/memes/" + data[str(i)]["id"] + ".png")
            images.append(process_image(img))
            img.close()
        except FileNotFoundError:
            try:
                img = PIL.Image.open("../data/memes/" + data[str(i)]["id"] + ".jpg")
                images.append(process_image(img))
                img.close()
            except FileNotFoundError:
                try:
                    img = url_to_image(data[str(i)]["media"])
                    images.append(process_image(img))
                    img.close()
                except:
                    print(str(i) + " depreciated")
                    data[str(i)] = "Null"

    return images, get_upvotes(data, max)


def process_image(img):
    img = tf.keras.preprocessing.image.img_to_array(img.convert('RGB'), data_format=None, dtype=None).astype(np.uint8)
    img = tf.image.resize(img, (512, 512)) / 255
    return img


def get_upvotes(data, max):
    print("Getting upvotes...")
    upvotes = []
    for i in range(1, max+1):
      if data[str(i)] != "Null":
        upvotes.append(data[str(i)]["ups"] - data[str(i)]["downs"])

    first, median, third = np.percentile(upvotes, 25), np.percentile(upvotes, 50), np.percentile(upvotes, 75)

    for i in range(len(upvotes)):
      if upvotes[i] < first:
          upvotes[i] = 0
      elif upvotes[i] < median:
          upvotes[i] = 1
      elif upvotes[i] < third:
          upvotes[i] = 2
      else:
          upvotes[i] = 3

    return upvotes


def divide(data):
    size = len(data)
    training_size = int(size * 0.8)
    test_size = int(size)
    return data[0:training_size], data[training_size: test_size]


def process(max):
    print("\nLoading data...\n")

    response = requests.get("https://raw.githubusercontent.com/RobinLmn/ML-MemePopularity/main/data/db.json")
    data = response.json()["_default"]

    images, upvotes = get_data(data, max)

    assert(len(images) == len(upvotes))

    print("\nSaving data...\n")
    np.save("images.npy", images)
    np.save("upvotes.npy", upvotes)

    print("\nFinished processing\n")


def update_upvotes():
    response = requests.get("https://raw.githubusercontent.com/RobinLmn/ML-MemePopularity/main/data/db.json")
    data = response.json()["_default"]
    np.save("upvotes.npy", get_upvotes(data, 3225))


def get(max):
    images, labels = np.load("../data/preprocessing/images.npy"), np.load("../data/preprocessing/upvotes.npy")
    assert(len(images) == len(labels))
    return images, labels


if __name__ == '__main__':
     #process(3225)
     update_upvotes()
