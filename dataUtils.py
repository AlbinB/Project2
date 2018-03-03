

import random
import tensorflow as tf
import os
import shutil
import sys
import tarfile
import numpy as np

from six.moves import urllib

# Global constants describing the CIFAR-10 data set.
DATA_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'
NUM_CLASSES = 10
TRAINING_FILENAMES = None
TEST_FILENAMES = None


def getCIFAR10Batch(is_eval, batch_size):
    """Get batch of CIFAR data
    Args:
    eval_data: bool, indicating if one should use the train or eval data set.
    Returns:
    images: Images. 4D numpy array of [batch_size, 32, 32, 3] size.
    labels: Labels. 1D numpy array of [batch_size] size.
    """

    global TRAINING_FILENAMES
    global TEST_FILENAMES

    dest_directory = "data"
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
                                                           float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    extracted_dir_path = os.path.join(dest_directory, 'cifar-10-batches-bin')
    if not os.path.exists(extracted_dir_path):
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)

    if not TRAINING_FILENAMES or not TEST_FILENAMES:
        data_dir = os.path.join(dest_directory, 'cifar-10-batches-bin')

        TRAINING_FILENAMES = [os.path.join(data_dir, 'data_batch_%d.bin' % i)
                         for i in range(1, 6)]
        TEST_FILENAMES = [os.path.join(data_dir, 'test_batch.bin')]

        for f in TRAINING_FILENAMES:
            if not tf.gfile.Exists(f):
                raise ValueError('Failed to find file: ' + f)

        for f in TEST_FILENAMES:
            if not tf.gfile.Exists(f):
                raise ValueError('Failed to find file: ' + f)


    batch_records = []
    batch_labels = []
    for i in range(batch_size):
        # Dimensions of the images in the CIFAR-10 dataset.
        # See http://www.cs.toronto.edu/~kriz/cifar.html for a description of the
        # input format.
        label_byte_count = 1  # 2 for CIFAR-100
        height = 32
        width = 32
        depth = 3
        image_byte_count = height * width * depth
        # Every record consists of a label followed by the image, with a
        # fixed number of bytes for each.
        file_record_count = 1000
        file_record_length = 3073

        if is_eval:
            filenames = TEST_FILENAMES
        else:
            filenames = TRAINING_FILENAMES

        with open(random.choice(filenames), "rb") as file:
            file.seek((random.randint(0,1000) % file_record_count) * file_record_length)
            label_bytes = file.read(label_byte_count)
            record_bytes = file.read(image_byte_count)

        label = int.from_bytes(label_bytes, byteorder="big")
        record = np.frombuffer(record_bytes, dtype=np.uint8)
        record = np.reshape(record, [depth, height, width])
        record = np.transpose(record, [1, 2, 0])

        batch_records.append(record)
        batch_labels.append(label)

    return batch_records, batch_labels


def accuracy(logits, labels):
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    return accuracy


def deleteDirectory(foldername):
    if os.path.exists(foldername) and os.path.isdir(foldername):
        print("Clearing folder {}".format(foldername))
        shutil.rmtree(foldername)
