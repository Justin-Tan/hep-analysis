#!/usr/bin/env python3
"""
Reads in TFRecords using TF Dataset Structure
"""
import tensorflow as tf
import argparse, glob, os, time

def dataset_input(dataDirectory, batchSize, numEpochs, fSize):
    filenames = glob.glob('{}/part*'.format(dataDirectory))
    dataset = tf.contrib.data.TFRecordDataset(filenames)

    # Extract data from `tf.Example` protocol buffer
    def parser(record, batchSize=128):
        keys_to_features = {
            "features": tf.FixedLenFeature([fSize], tf.float32),
            "labels": tf.FixedLenFeature((), tf.float32,
            default_value=tf.zeros([], dtype=tf.float32)),
        }
        parsed = tf.parse_single_example(record, keys_to_features)
        label = tf.cast(parsed['labels'], tf.int16)

        return parsed['features'], label

    # Transform into feature, label tensor pair
    dataset = dataset.map(parser)
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(batchSize)
    dataset = dataset.repeat(numEpochs)

    return dataset

trainDataset = dataset_input(trainPath, batchSize=128, numEpochs=64, fSize=44)
testDataset = dataset_input(testPath, batchSize=128, numEpochs=64, fSize=44)
iterator = tf.contrib.data.Iterator.from_structure(trainDataset.output_types,
                                   trainDataset.output_shapes)

train_init_op = iterator.make_initializer(trainDataset)
test_init_op = iterator.make_initializer(testDataset)

next_example, next_label = iterator.get_next()


def dataset_input_fn(dataDirectory):
  filenames = ["/var/data/file1.tfrecord", "/var/data/file2.tfrecord"]
  dataset = tf.contrib.data.TFRecordDataset(filenames)

  # Use `tf.parse_single_example()` to extract data from a `tf.Example`
  # protocol buffer, and perform any additional per-record preprocessing.
  def parser(record):
    keys_to_features = {
        "image_data": tf.FixedLenFeature((), tf.string, default_value=""),
        "date_time": tf.FixedLenFeature((), tf.int64, default_value=""),
        "label": tf.FixedLenFeature((), tf.int64,
                                    default_value=tf.zeros([], dtype=tf.int64)),
    }
    parsed = tf.parse_single_example(record, keys_to_features)

    # Perform additional preprocessing on the parsed data.
    image = tf.decode_jpeg(parsed["image_data"])
    image = tf.reshape(image, [299, 299, 1])
    label = tf.cast(parsed["label"], tf.int32)

    return {"image_data": image, "date_time": parsed["date_time"]}, label

  # Use `Dataset.map()` to build a pair of a feature dictionary and a label
  # tensor for each example.
  dataset = dataset.map(parser)
  dataset = dataset.shuffle(buffer_size=10000)
  dataset = dataset.batch(32)
  dataset = dataset.repeat(num_epochs)
  iterator = dataset.make_one_shot_iterator()

  # `features` is a dictionary in which each value is a batch of values for
  # that feature; `labels` is a batch of labels.
  features, labels = iterator.get_next()
  return features, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('tfrecords', help='Path to tfrecords directory')
    args = parser.parse_args()

    t0 = time.time()
    # Stuff
    print('Read time: {}'.format(time.time() - t0))

if __name__ == "__main__":
    main()
