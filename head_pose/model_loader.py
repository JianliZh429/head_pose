import os

import numpy as np

BASE_DIR = os.path.dirname(__file__)


def get_full_model_points(filename=os.path.join(BASE_DIR, '../models/model_3D_68.txt')):
    """Get all 68 3D model points from file"""
    raw_value = []
    with open(filename) as file:
        for line in file:
            raw_value.append(line)
    model_points = np.array(raw_value, dtype=np.float32)
    model_points = np.reshape(model_points, (3, -1)).T
    # model_points *= 4
    model_points[:, -1] *= -1

    return model_points


def get_nose_eye_chin_mouth_6():
    """
    :return: nose, chin, right eye corner, left eye corner, right mouth corner, left mouth corner
    """
    model_points = get_full_model_points()
    return np.array([model_points[30], model_points[8], model_points[36],
                     model_points[45], model_points[48], model_points[54]])


def get_nose_eye_chin_mouth_2d(landmarks):
    return np.array([landmarks[30], landmarks[8], landmarks[36], landmarks[45], landmarks[48], landmarks[54]],
                    dtype=np.double)
