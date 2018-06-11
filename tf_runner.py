import argparse
import json
import os
import time

import matplotlib.pyplot as plt
import tensorflow as tf

import datasource as ds
import numpy as np

IMAGE_WIDTH = 200
IMAGE_HEIGHT = 200
IMAGE_CHANNEL = 3

N_AGE_CLASSES = 8
N_TOTAL_LABELS = N_AGE_CLASSES + 1

EPOCH_LENGTH = 100
BATCH_SIZE = 50
LEARNING_RATE = 0.01

RANDOM_SEED = 1

DEFAULT_OUTPUT_PATH: str = os.curdir + os.sep + 'model'


def main():
    # argument_parser = argparse.ArgumentParser()
    # # Model file
    # argument_parser.add_argument('model', help='path to model object')
    # group = argument_parser.add_mutually_exclusive_group(required=True)
    # group.add_argument(
    #     '-tr', '--train', type=argparse.FileType('r'), help='trains ')
    # group.add_argument('-ts', '--test', type=argparse.FileType('r'))
    # argument_parser.add_argument(
    #     '-s', '--save', nargs='?', const=DEFAULT_OUTPUT_PATH, type=str)
    # arguments = vars(argument_parser.parse_args())

    # Create given output folder if it doesn't exist
    # if not os.path.isdir(arguments['save']):
    #     os.mkdir(arguments['save'])

    input_placeholder: tf.Tensor = tf.placeholder(
        tf.float32, shape=[None, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNEL])

    output_placeholder: tf.Tensor = tf.placeholder(
        tf.float32, shape=[None, N_TOTAL_LABELS])

    layer_conv_1: tf.Tensor = new_conv_layer(
        input_tensor=input_placeholder,
        num_input_channels=IMAGE_CHANNEL,
        filter_size=5,
        num_filters=8,
    )

    layer_conv_1_pooled: tf.Tensor = new_pooling(layer_conv_1, 2)

    layer_conv_2: tf.Tensor = new_conv_layer(
        input_tensor=layer_conv_1_pooled,
        num_input_channels=8,
        filter_size=3,
        num_filters=16,
    )
    layer_conv_2_pooled: tf.Tensor = new_pooling(layer_conv_2, 2)

    layer_conv_3: tf.Tensor = new_conv_layer(
        input_tensor=layer_conv_2_pooled,
        num_input_channels=16,
        filter_size=3,
        num_filters=16,
    )

    layer_conv_3_pooled = new_pooling(layer_conv_3, 2)

    layer_flat, num_features = flatten_layer(layer_conv_3_pooled)

    layer_fc_1: tf.Tensor = new_fc_layer(
        input_tensor=layer_flat, num_inputs=num_features, num_outputs=256)

    layer_fc_1_activated: tf.Tensor = tf.nn.sigmoid(layer_fc_1)

    layer_fc_2: tf.Tensor = new_fc_layer(
        input_tensor=layer_fc_1_activated, num_inputs=256, num_outputs=32)

    layer_fc_2_activated: tf.Tensor = tf.nn.sigmoid(layer_fc_2)

    layer_output: tf.Tensor = new_fc_layer(
        input_tensor=layer_fc_2_activated, num_inputs=32, num_outputs=N_TOTAL_LABELS)

    layer_output_activated: tf.Tensor = tf.nn.sigmoid(layer_output)

    cost: tf.Tensor = tf.losses.log_loss(
        output_placeholder, layer_output_activated)

    optimizer: tf.Operation = tf.train.AdamOptimizer(
        LEARNING_RATE).minimize(cost)

    predictions = tf.round(layer_output_activated)
    age_pred, gender_pred = tf.split(
        predictions, [8, 1], axis=1)

    age_labels, gender_labels = tf.split(
        output_placeholder, [8, 1], axis=1)

    age_pred_equalities = tf.equal(tf.argmax(
        age_pred, 1), tf.argmax(age_labels, 1))

    print(age_pred_equalities.shape)

    age_accuracy: tf.Tensor = tf.reduce_mean(
        tf.cast(age_pred_equalities, tf.float32), axis=0)

    gender_pred_equalities: tf.Tensor = tf.equal(gender_pred, gender_labels)
    gender_accuracy: tf.Tensor = tf.reduce_mean(
        tf.cast(gender_pred_equalities, tf.float32))

    init_g = tf.global_variables_initializer()
    init_l = tf.local_variables_initializer()

    with tf.Session() as sess:
        sess.run(init_g)
        sess.run(init_l)

        saver = tf.train.Saver()

        training_losses = []
        validation_losses = []

        for training_batch_images, training_batch_labels, training_batch_index \
                in ds.training_batch_generator(BATCH_SIZE):

            print("Starting batch {}\n".format(training_batch_index + 1))

            for current_epoch in range(EPOCH_LENGTH):

                feed = {
                    input_placeholder: training_batch_images,
                    output_placeholder: training_batch_labels
                }

                (
                    training_epoch_accuracy_age,
                    training_epoch_accuracy_gender,
                    training_epoch_cost,
                    _
                ) = sess.run(
                    [age_accuracy, gender_accuracy, cost, optimizer], feed_dict=feed)

                epoch_log_message = "Batch {:3}, Epoch {:3} -> Cost: {}, Age accuracy: {:3.1%}, Gender accuracy: {:3.1%}"
                epoch_log_message = epoch_log_message.format(training_batch_index + 1,
                                                             current_epoch + 1,
                                                             training_epoch_cost,
                                                             training_epoch_accuracy_age,
                                                             training_epoch_accuracy_gender)
                print(epoch_log_message)

            training_losses.append(training_epoch_cost)

            print("Batch {} training has been finished!\n",
                  training_batch_index + 1)
            print("Testing with validaiton dataset...\n")

            validation_total_accuracy_age = 0
            validation_total_accuracy_gender = 0
            validation_total_cost = 0

            for validation_batch_images, validation_batch_labels, validation_batch_index\
                    in ds.validation_batch_generator(BATCH_SIZE):

                print('testing val batch {}'.format(validation_batch_index + 1))

                feed = {
                    input_placeholder: validation_batch_images,
                    output_placeholder: validation_batch_labels
                }

                (
                    validation_batch_accuracy_age,
                    validation_batch_accuracy_gender,
                    validation_batch_cost
                ) = sess.run([age_accuracy, gender_accuracy, cost], feed_dict=feed)

                validation_total_accuracy_age += validation_batch_accuracy_age
                validation_total_accuracy_gender += validation_batch_accuracy_gender
                validation_total_cost += validation_batch_cost


            validation_average_accuracy_age = \
                validation_total_accuracy_age / (validation_batch_index + 1)
            validation_average_accuracy_gender = \
                validation_total_accuracy_gender / (validation_batch_index + 1)
            validation_average_cost = validation_total_cost / (validation_batch_index + 1)

            val_log_str = 'Validation: \n Cost: {}, Age accuracy: {:3.1%}, Gender accuracy: {:3.1%}'
            print(val_log_str.format(validation_average_cost,
                                     validation_average_accuracy_age,
                                     validation_average_accuracy_gender))

            validation_losses.append(validation_average_cost)
            validation_average_accuracy = \
                (validation_average_accuracy_age +
                 validation_average_accuracy_gender) / 2

            json_obj = {
                'training': np.array(training_losses).tolist(),
                'validation': np.array(validation_losses).tolist(),
                'accuracy': np.asscalar(validation_average_accuracy)
            }

            with open('result.json', 'w') as outfile:
                json.dump(json_obj, outfile, indent=4)

            save_path = saver.save(sess, "/model/checkpoint_110618_042900")
            print("Model saved in path: %s" % save_path)

        print("\nTraining finished at {}!\n".format(time.asctime()))

        plt.plot(training_batch_index + 1, training_losses)
        plt.plot(training_batch_index + 1, validation_losses)

        plt.title("Learning")
        plt.legend(['Training Loss', 'Validation Loss'], loc='upper left')

        plt.show()


def new_weights(shape):
    randomized_tensor = tf.random_normal(
        shape=shape, dtype=tf.float32, seed=RANDOM_SEED)
    weight = tf.Variable(randomized_tensor)
    return weight


def new_biases(length):
    randomized_tensor = tf.random_normal(
        shape=[length], dtype=tf.float32, seed=(RANDOM_SEED + 1))
    bias = tf.Variable(randomized_tensor)
    return bias


def new_conv_layer(input_tensor: tf.Tensor, num_input_channels: int, filter_size: int, num_filters: int):
    shape = [filter_size, filter_size, num_input_channels, num_filters]
    weights = new_weights(shape=shape)
    biases = new_biases(length=num_filters)

    layer = tf.nn.conv2d(input=input_tensor, filter=weights,
                         strides=[1, 1, 1, 1], padding='SAME')

    layer = tf.add(layer, biases)

    layer = tf.nn.relu(layer)

    return layer


def new_pooling(input_tensor: tf.Tensor, size: int):
    return tf.nn.max_pool(value=input_tensor, ksize=[1, size, size, 1],
                          strides=[1, size, size, 1], padding='SAME')


def flatten_layer(layer: tf.Tensor):
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()
    layer_flat = tf.reshape(layer, [-1, num_features])
    return layer_flat, num_features


def new_fc_layer(input_tensor: tf.Tensor, num_inputs: int, num_outputs: int):
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)
    layer = tf.add(tf.matmul(input_tensor, weights), biases)
    return layer


if __name__ == '__main__':
    main()
