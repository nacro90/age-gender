import json
import os
from random import shuffle

import numpy as np
from PIL import Image

IMAGE_WIDTH = 200
IMAGE_HEIGHT = 200
IMAGE_CHANNEL = 3

DATASET_PATH = os.curdir + os.sep + 'data' + os.sep + 'Adience'
IMAGE_FOLDER = os.sep + 'faces'
ANNOTATION_FOLDER = os.sep + 'annotations'
JSON_FOLDER = os.sep + 'set'

ANNOTATION_FILE_TEMPLATE = 'fold_{}_data.txt'

JSON_NAME_TRAIN = 'train.json'
JSON_NAME_VALIDATION = 'validation.json'
JSON_NAME_TEST = 'test.json'


def training_batch_generator(batch_size):
    train_data = json.load(
        open(DATASET_PATH + JSON_FOLDER + os.sep + JSON_NAME_TRAIN))

    n_batches = len(train_data) // batch_size

    for batch_index in range(n_batches):

        images, labels = parse_json(train_data, batch_size, batch_index)

        yield images, labels, batch_index




def validation_batch_generator(batch_size):
    validation_data = json.load(
        open(DATASET_PATH + JSON_FOLDER + os.sep + JSON_NAME_VALIDATION))

    n_batches = len(validation_data) // batch_size

    for batch_index in range(n_batches):

        images, labels = parse_json(validation_data, batch_size, batch_index)

        yield images, labels, batch_index


def test_batch_generator(batch_size):
    test_data = json.load(
        open(DATASET_PATH + JSON_FOLDER + os.sep + JSON_NAME_TEST))

    n_batches = len(test_data) // batch_size

    for batch_index in range(n_batches):

        images, labels = parse_json(test_data, batch_size, batch_index)

        yield images, labels, batch_index


def parse_json(train_data, batch_size, batch_index):
    images = np.zeros((batch_size, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNEL))
    labels = np.zeros((batch_size, 9))

    for i, train_row in enumerate(train_data[batch_index * batch_size:(batch_index + 1) * batch_size]):

        image_path = \
            DATASET_PATH + \
            IMAGE_FOLDER + os.sep + \
            train_row['folder_name'] + os.sep + \
            'coarse_tilt_aligned_face' + '.' + \
            str(train_row['face_id']) + '.' + \
            train_row['image_file_name']

        image = Image.open(image_path)
        image_np = np.array(image.getdata(), dtype=np.float32)
        image_np = image_np.reshape(
            IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNEL)

        images[i] = image_np

        labels[i][train_row['age_class']] = 1
        labels[i][8] = train_row['gender_class']

    return images, labels


def main():
    for images, labels, batch_index in training_batch_generator(1):
        print(images)
        print(labels)
        if batch_index == 0:
            break


if __name__ == '__main__':
    main()
