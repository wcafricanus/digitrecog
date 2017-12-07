import gzip

import pickle

import os

import cv2

import numpy as np


def load_data(filename):
    f = gzip.open(filename, 'rb')
    training_data, validation_data, test_data = pickle.load(f)
    f.close()
    return [training_data, validation_data, test_data]


dir_name = os.path.dirname(__file__)

file_path = dir_name + "/mnist_expanded.pkl.gz"
training_data, validation_data, test_data = load_data(file_path)
for image in training_data[0]:
    cv2.imshow('image', np.reshape(image, (-1, 28)))
    cv2.waitKey()
    cv2.destroyAllWindows()

