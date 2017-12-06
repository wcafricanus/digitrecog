import json
import os

import cv2
import numpy as np
import pytest

from preprocess import strip_label, draw_greyscale_digit, preprocess_data, preprocess_single_data, rescale_image
from loadmodeltest import Evaluator
from imageprocess.elasticdistortion import elastic_transform


@pytest.fixture
def evaluator():
    e = Evaluator()
    return e


@pytest.fixture
def test_dict():
    cwd = os.path.dirname(__file__)
    with open(cwd+'/test3.txt', 'r') as myfile:
        data = json.load(myfile)
    vas_cog_block = data['test']['vasCogBlock']
    vas_block_size = data['test']['vasBlockSize']
    return preprocess_data(vas_cog_block, vas_block_size)


@pytest.fixture
def single_item_dict():
    cwd = os.path.dirname(__file__)
    with open(cwd + '/mark_one_body.txt', 'r') as myfile:
        data = json.load(myfile)
    vas_cog_block = data['vasCogBlock']
    vas_block_size = data['vasBlockSize']
    return preprocess_single_data(vas_cog_block, vas_block_size)


def test_evaluator(evaluator, test_dict):
    result = evaluator.evaluate(test_dict)
    pass


def test_rescale(test_dict):
    images_2d = [value[0].reshape(28, 28) if value[0] is not None else np.zeros((28, 28)) for value in
                 test_dict.values()]
    for image in images_2d:
        distorted = rescale_image(image)

        shape1 = image.shape
        shape2 = distorted.shape
        combined = np.zeros((shape1[0], shape1[1] + shape2[1]), np.float32)
        combined[:,:shape1[1]] = image
        combined[:,shape1[1]:] = distorted

        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('image', 100, 100)
        cv2.imshow('image', combined)
        cv2.waitKey()
        cv2.destroyAllWindows()


def test_elastic_distortion(test_dict):
    images_2d = [value[0].reshape(28, 28) if value[0] is not None else np.zeros((28, 28)) for value in
                 test_dict.values()]
    for image in images_2d:
        distorted = elastic_transform(image, alpha=8, sigma=3)

        shape1 = image.shape
        shape2 = distorted.shape
        combined = np.zeros((shape1[0], shape1[1] + shape2[1]), np.float32)
        combined[:,:shape1[1]] = image
        combined[:,shape1[1]:] = distorted

        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('image', 100, 100)
        cv2.imshow('image', combined)
        cv2.waitKey()
        cv2.destroyAllWindows()