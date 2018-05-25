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


def _nose_chin_2eyes_2mouth(landmarks):
    """
    :return: nose, chin, right eye corner, left eye corner, right mouth corner, left mouth corner
    """

    return np.array([landmarks[30], landmarks[8], landmarks[36],
                     landmarks[45], landmarks[48], landmarks[54]], dtype=np.double)


def _nose_2eyes_2ears(landmarks):
    return np.array([landmarks[30], landmarks[40], landmarks[47],
                     landmarks[1], landmarks[15]], dtype=np.double)


def get_model_3d_points(mode='nose_chin_eyes_mouth'):
    """
    :param mode: must in ['nose_eyes_ears','nose_chin_eyes_mouth']
    :return:
    """
    model_points = get_full_model_points()
    if mode == 'nose_chin_eyes_mouth':
        return _nose_chin_2eyes_2mouth(model_points)
    elif mode == 'nose_eyes_ears':
        return _nose_2eyes_2ears(model_points)
    else:
        raise ValueError('mode must be either "nose_eyes_ears" or "nose_chin_eyes_mouth" ')


def get_points_from_landmarks(landmarks, mode='nose_chin_eyes_mouth'):
    """
    :param landmarks: 68 points of face landmark
    :param mode: :param mode: must in ['nose_eyes_ears','nose_chin_eyes_mouth']
    :return:
    """
    if mode == 'nose_chin_eyes_mouth':
        return _nose_chin_2eyes_2mouth(landmarks)
    elif mode == 'nose_eyes_ears':
        return _nose_2eyes_2ears(landmarks)
    else:
        raise ValueError('mode must be either "nose_eyes_ears" or "nose_chin_eyes_mouth" ')
