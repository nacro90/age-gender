import os

from PIL import Image


DATASET_PATH = os.curdir + os.sep + 'data' + os.sep + 'Adience'
IMAGE_FOLDER = os.sep + 'faces'
ANNOTATION_FOLDER = os.sep + 'annotations'
JSON_FOLDER = os.sep + 'set'


def main():

    path = DATASET_PATH + IMAGE_FOLDER

    directories = os.listdir(DATASET_PATH + IMAGE_FOLDER)

    counter = 0

    for d in directories:
        directory_path = path + os.sep + d
        image_file_names = os.listdir(directory_path)
        for i in image_file_names:
            if i[-3:] != 'txt':
                image_path = directory_path + os.sep + i
                image: Image.Image = Image.open(image_path)
                resized = image.resize((200, 200))
                resized.save(image_path)
                counter += 1
                print(counter)

if __name__ == '__main__':
    main()

# 464 467
