import os
import json
import re

DATASET_PATH = os.curdir + os.sep + 'data' + os.sep + 'Adience'
IMAGE_FOLDER = os.sep + 'faces'
ANNOTATION_FOLDER = os.sep + 'annotations'
JSON_FOLDER = os.sep + 'set'

ANNOTATION_FILE_TEMPLATE = 'fold_{}_data.txt'

JSON_NAME_TRAIN = 'train.json'
JSON_NAME_VALIDATION = 'validation.json'
JSON_NAME_TEST = 'test.json'

# 0 1 2 for train 3 for cross validation 4 for test


def main():

    # Train
    people = []

    for i in range(3):
        file_path = DATASET_PATH + ANNOTATION_FOLDER + os.sep + \
            ANNOTATION_FILE_TEMPLATE.format(str(i))

        with open(file_path, 'r') as file:

            # Skip first line
            file.readline()

            for line in file:
                person = parse_line_to_person(line)
                people.append(person)

    json_out = json.dumps(people, indent=4, sort_keys=True)

    with open(DATASET_PATH + JSON_FOLDER + os.sep + JSON_NAME_TRAIN, 'w') as out_file:
        out_file.write(json_out)

    # Validation
    people = []

    file_path = DATASET_PATH + ANNOTATION_FOLDER + os.sep + \
        ANNOTATION_FILE_TEMPLATE.format(3)

    with open(file_path, 'r') as file:

        # Skip first line
        file.readline()

        for line in file:
            person = parse_line_to_person(line)
            people.append(person)

    json_out = json.dumps(people, indent=4, sort_keys=True)

    with open(DATASET_PATH + JSON_FOLDER + os.sep + JSON_NAME_VALIDATION, 'w') as out_file:
        out_file.write(json_out)

    # Test
    people = []

    file_path = DATASET_PATH + ANNOTATION_FOLDER + os.sep + \
        ANNOTATION_FILE_TEMPLATE.format(4)

    with open(file_path, 'r') as file:

        # Skip first line
        file.readline()

        for line in file:
            person = parse_line_to_person(line)
            people.append(person)

    json_out = json.dumps(people, indent=4, sort_keys=True)

    with open(DATASET_PATH + JSON_FOLDER + os.sep + JSON_NAME_TEST, 'w') as out_file:
        out_file.write(json_out)


def parse_line_to_person(line):
    splitted_line = re.split(r'\s+', line)

    # '114841417@N06'
    folder_name = splitted_line[0]

    # '12068804204_085d553238_o.jpg'
    image_file_name = splitted_line[1]

    face_id = int(splitted_line[2])

    # '(60, 100)'
    raw_age = splitted_line[3] + ' ' + splitted_line[4]


    # Age class 0-7
    age_class = None
    if raw_age[1:3] == '60':
        age_class = 7
    elif raw_age[1:3] == '48':
        age_class = 6
    elif raw_age[1:3] == '38':
        age_class = 5
    elif raw_age[1:3] == '25':
        age_class = 4
    elif raw_age[1:3] == '15':
        age_class = 3
    elif raw_age[1] == '8':
        age_class = 2
    elif raw_age[1] == '4':
        age_class = 1
    elif raw_age[1] == '0':
        age_class = 0

    raw_gender = splitted_line[5]

    gender_class = 0 if raw_gender == 'm' else 1

    return {
        'folder_name': folder_name,
        'image_file_name': image_file_name,
        'age_raw': raw_age,
        'age_class': age_class,
        'gender_raw': raw_gender,
        'gender_class': gender_class,
        'face_id': face_id
    }


if __name__ == '__main__':
    main()
