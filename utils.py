import cv2
import numpy as np
from matplotlib import pyplot as plt


def trans_mat(tx, ty):
    return np.array([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1],
    ])


def rot_mat(q):
    return np.array([
        [np.cos(q), -np.sin(q), 0],
        [np.sin(q), np.cos(q), 0],
        [0, 0, 1]
    ])


def shear_mat(shx, shy):
    return np.array([
        [1, shx, 0],
        [shy, 1, 0],
        [0, 0, 1]
    ])


def scale_mat(sx, sy):
    return np.array([
        [sx, 0, 0],
        [0, sy, 0],
        [0, 0, 1]
    ])


def get_transformation_matrix(center, tx, ty, sx, sy, shx, shy, q):
    move_to_center = trans_mat(tx=center[0], ty=center[1])

    rotate = rot_mat(q)

    translate = trans_mat(tx, ty)

    scale = scale_mat(sx, sy)

    shear = shear_mat(shx, shy)

    move_back = trans_mat(tx=-center[0], ty=-center[1])

    matrix = move_to_center @ rotate @ translate @ scale @ shear @ move_back
    # matrix_inv = np.linalg.inv(matrix)

    return matrix


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
