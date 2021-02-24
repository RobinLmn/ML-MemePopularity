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
    return PIL.Image.open(BytesIO(response.content)).convert('RGB')


def get_data(data, max):

    print("Getting images...")
    images = []
    for i in range(1, max+1):
        percent = round(i * 100 / max, 2)
        print(str(percent) + "%", end="\r")
        try:
            img = PIL.Image.open("../data/memes/" + data[str(i)]["id"] + ".png").convert('RGB')
            images.append(process_image(img))
            img.close()
        except FileNotFoundError:
            try:
                img = PIL.Image.open("../data/memes/" + data[str(i)]["id"] + ".jpg").convert('RGB')
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

    print("Getting upvotes...")
    upvotes = []
    for i in range(1, max+1):
      if data[str(i)] != "Null":
        upvotes.append(data[str(i)]["ups"] - data[str(i)]["downs"])
    mean = np.mean(upvotes)

    for i in range(len(upvotes)):
      upvotes[i] = 0 if upvotes[i] >= mean else 1

    return images, upvotes


def process_image(img):
    img = tf.keras.preprocessing.image.img_to_array(img, data_format=None, dtype=None).astype(np.uint8)
    img = tf.image.resize(img, (1028, 1028)) / 255
    return img

def divide(data):
    size = len(data)
    training_size = int(size * 0.8)
    test_size = int(size)
    return data[0:training_size], data[training_size: test_size]

def get(max):
    print("\nLoading data...\n")

    response = requests.get("https://raw.githubusercontent.com/RobinLmn/ML-MemePopularity/main/data/db.json")
    data = response.json()["_default"]

    images, upvotes = get_data(data, max)

    assert(len(images) == len(upvotes))

    print("\nFinished processing\n")

    return divide(images), divide(upvotes)


def main(max):
    images, labels = get(max)
    train_images, test_images = images
    train_labels, test_labels = labels

    return train_images, test_images, train_labels, test_labels


if __name__ == '__main__':
     train_images, test_images, train_labels, test_labels = main(10)

     print(train_labels)
     plt.imshow(train_images[0])
     plt.show()
