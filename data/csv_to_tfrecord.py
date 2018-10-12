from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import random

import tensorflow as tf

FEATURE_NAME = "x"
LABEL_NAME = "y"


def initialise_flags(args_parser):
    args_parser.add_argument("--train-csv", required=True)
    args_parser.add_argument("--eval-csv", required=True)
    args_parser.add_argument("--train-tfrecord", required=True)
    args_parser.add_argument("--eval-tfrecord", required=True)

    return args_parser.parse_args()

def eval_csv_row_to_example(row):

    feature_dict = {
        'pixels': tf.train.Feature(int64_list=tf.train.Int64_list(value=row)),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature_dict))


def train_csv_row_to_example(row):
    x_raw, y_raw = row.split(",")
    x, y = float(x_raw.strip()), float(y_raw.strip())
    feature_dict = {
        FEATURE_NAME: tf.train.Feature(float_list=tf.train.FloatList(value=[x])),
        LABEL_NAME: tf.train.Feature(float_list=tf.train.FloatList(value=[y])),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature_dict))


def main():
    train_csv_path = os.path.relpath(FLAGS.train_csv)
    eval_csv_path = os.path.relpath(FLAGS.train_csv)
    train_tfrecord_path = os.path.relpath(FLAGS.train_tfrecord)
    eval_tfrecord_path = os.path.relpath(FLAGS.eval_tfrecord)
    with open(train_csv_path, "rb") as train_csv:
        next(train_csv)  # Skip heading row.
        writer = tf.python_io.TFRecordWriter(train_tfrecord_path)
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            example = train_csv_row_to_example(row)
            writer.write(example.SerializeToString())
        
        eval_writer = tf.python_io.TFRecordWriter(eval_tfrecord_path)
        validation_writer = tf.python_io.TFRecordWriter(validation_tfrecord_path)
        for row in csv:
            example = csv_row_to_example(row)
            rand = random.random()
            if rand < 0.8:
                train_writer.write(example.SerializeToString())
            elif 0.8 <= rand < 0.9:
                eval_writer.write(example.SerializeToString())
            else:
                validation_writer.write(example.SerializeToString())

        train_writer.close()
        eval_writer.close()
        validation_writer.close()


args_parser = argparse.ArgumentParser()
FLAGS = initialise_flags(args_parser)

if __name__ == "__main__":
    main()
