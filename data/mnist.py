import gzip

import pickle

import os

import numpy as np


def load_data(filename):
    f = gzip.open(filename, 'rb')
    training_data, validation_data, test_data = pickle.load(f)
    f.close()
    return [training_data, validation_data, test_data]

class DataSet():
    def __init__(self, data):
        self._images = data[0]
        self._labels = data[1]
        self.data_length = len(self._images)
        self.cursor = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    def next_batch(self, batch_size, shuffle=True):
        start = self.cursor
        if batch_size > self.data_length:
            raise OverflowError("Batch size bigger than number of total entries.")
        if start+batch_size > self.data_length:
            # Get the rest examples in this epoch
            rest_num_examples = self.data_length - start
            images_rest_part = self._images[start:self.data_length]
            labels_rest_part = self._labels[start:self.data_length]
            # Shuffle the data
            if shuffle:
                perm = np.arange(self.data_length)
                np.random.shuffle(perm)
                self._images = self.images[perm]
                self._labels = self.labels[perm]
            # Start next epoch
            start = 0
            self.cursor = batch_size - rest_num_examples
            end = self.cursor
            images_new_part = self._images[start:end]
            labels_new_part = self._labels[start:end]
            return np.concatenate((images_rest_part, images_new_part), axis=0), np.concatenate(
                (labels_rest_part, labels_new_part), axis=0)
        else:
            self.cursor += batch_size
            end = self.cursor
            return self._images[start:end], self._labels[start:end]


class Mnist():
    def __init__(self):
        dir_name = os.path.dirname(__file__)
        file_path = dir_name + "/mnist_expanded.pkl.gz"
        training_data, validation_data, test_data = load_data(file_path)
        self.train = DataSet(training_data)
        self.validation = DataSet(validation_data)
        self.test = DataSet(test_data)
