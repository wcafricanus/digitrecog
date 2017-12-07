import cv2
import numpy as np


def affine_distort(image):
    rows, cols = image.shape
    pts1 = np.float32([[4, 4], [8, 4], [4, 8]])
    pts2 = np.float32([[4, 6], [8, 6], [4, 12]])
    M = cv2.getAffineTransform(pts1, pts2)

    dst = cv2.warpAffine(image, M, (cols, rows*3//4))

    pts3 = np.float32([[4, 4], [8, 4], [4, 8]])
    pts4 = np.float32([[4, 2], [8, 2], [4, 4]])
    M = cv2.getAffineTransform(pts3, pts4)

    dst2 = cv2.warpAffine(image[rows//2:, :], M, (cols, rows * 1 // 4))

    return dst2