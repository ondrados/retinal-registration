import cv2
import numpy as np
from matplotlib import pyplot as plt


def get_transformation_matrix(center, tx, ty, sx, sy, shx, shy, q):
    move_to_center = np.array([
        [1, 0, center[0]],
        [0, 1, center[1]],
        [0, 0, 1],
    ])

    rotate = np.array([
        [np.cos(q), -np.sin(q), 0],
        [np.sin(q), np.cos(q), 0],
        [0, 0, 1]
    ])

    translate = np.array([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1],
    ])

    scale = np.array([
        [sx, 0, 0],
        [0, sy, 0],
        [0, 0, 1]
    ])

    shear = np.array([
        [1, shy, 0],
        [shx, 1, 0],
        [0, 0, 1]
    ])

    move_back = np.array([
        [1, 0, -center[0]],
        [0, 1, -center[1]],
        [0, 0, 1],
    ])

    matrix = move_to_center @ rotate @ translate @ scale @ shear @ move_back
    matrix_inv = np.linalg.inv(matrix)

    return matrix_inv


def get_random_transformation(image):
    rows, cols = image.shape
    center = (image.shape[1] // 2, image.shape[0] // 2)
    tx = np.random.uniform(-50, 50, 1)[0]
    ty = np.random.uniform(-50, 50, 1)[0]
    sx = np.random.uniform(0.95, 1.05, 1)[0]
    sy = np.random.uniform(0.95, 1.05, 1)[0]
    shx = np.random.uniform(-0.1, 0.1, 1)[0]
    shy = np.random.uniform(-0.1, 0.1, 1)[0]
    q = np.radians(np.random.uniform(-10, 10, 1)[0])

    matrix = get_transformation_matrix(center, tx, ty, sx, sy, shx, shy, q)

    matrix_opencv = np.float32(matrix.flatten()[:6].reshape(2, 3))
    trans_image = cv2.warpAffine(image, matrix_opencv, (cols, rows))

    h = 500
    w = int(h * 1.5)
    image_crop = image[center[1] - h // 2:center[1] + h // 2, center[0] - w // 2:center[0] + w // 2]
    trans_image_crop = trans_image[center[1] - h // 2:center[1] + h // 2, center[0] - w // 2:center[0] + w // 2]

    return trans_image, trans_image_crop, image_crop, [tx, ty, sx, sy, shx, shy, q]




































