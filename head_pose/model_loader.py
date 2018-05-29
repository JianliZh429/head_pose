import os

import numpy as np
from pkg_resources import resource_filename

BASE_DIR = os.path.dirname(__file__)


def get_68_3d_model():
    return resource_filename(__name__, 'models/model_3D_68.txt')


def get_full_model_points(filename=get_68_3d_model()):
    """Get all 68 3D model points from file"""
    raw_value = []
    with open(filename) as file:
        for line in file:
            raw_value.append(line)
    model_points = np.array(raw_value, dtype=np.float32)
    model_points = np.reshape(model_points, (3, -1)).T
    model_points[:, -1] *= -1

    return model_points


def _nose_chin_2eyes_2mouth(landmarks):
    """
    :return: nose, chin, right eye corner, left eye corner, right mouth corner, left mouth corner
    """

    return np.array([_centroid_nose(landmarks), _centroid_chin(landmarks),
                     _centroid_right_eye(landmarks), _centroid_left_eye(landmarks),
                     _centroid_mouth_right_corner(landmarks), _centroid_mouth_left_corner(landmarks)],
                    dtype=np.double)


def _nose_2eyes_2ears(landmarks):
    return np.array([_centroid_nose(landmarks),
                     _centroid_right_eye(landmarks), _centroid_left_eye(landmarks),
                     _centroid_right_ear(landmarks), _centroid_left_ear(landmarks)],
                    dtype=np.double)


def _nose_2eyes_2mouth(landmarks):
    """
    :param landmarks:
    :return: nose, right eye corner, left eye corner, right mouth corner, left mouth corner
    """
    return np.array([_centroid_nose(landmarks),
                     _centroid_right_eye(landmarks), _centroid_left_eye(landmarks),
                     _centroid_mouth_right_corner(landmarks), _centroid_mouth_left_corner(landmarks)],
                    dtype=np.double)


def _centroid_chin(landmarks):
    chins = np.array([landmarks[7], landmarks[8], landmarks[9]])
    return _centroid(chins)


def _centroid_left_ear(landmarks):
    left_ear = np.array([landmarks[14], landmarks[15], landmarks[16]])
    return _centroid(left_ear)


def _centroid_right_ear(landmarks):
    right_ear = np.array([landmarks[0], landmarks[1], landmarks[2]])
    return _centroid(right_ear)


def _centroid_mouth_left_corner(landmarks):
    left_mouth = np.array([landmarks[53], landmarks[54], landmarks[55], landmarks[64]])
    return _centroid(left_mouth)


def _centroid_mouth_right_corner(landmarks):
    right_mouth = np.array([landmarks[48], landmarks[49], landmarks[60], landmarks[59]])
    return _centroid(right_mouth)


def _centroid_nose(landmarks):
    noses = np.array([landmarks[27], landmarks[28], landmarks[29], landmarks[30], landmarks[31],
                      landmarks[32], landmarks[33], landmarks[34], landmarks[35]])
    return _centroid(noses)


def _centroid_right_eye(landmarks):
    right_eyes = np.array([landmarks[36], landmarks[37], landmarks[38], landmarks[39], landmarks[40], landmarks[41]])
    return _centroid(right_eyes)


def _centroid_left_eye(landmarks):
    left_eyes = np.array([landmarks[42], landmarks[43], landmarks[44], landmarks[45], landmarks[46], landmarks[47]])
    return _centroid(left_eyes)


def _centroid(points):
    xs = points[:, 0]
    ys = points[:, 1]

    if len(points[0]) == 3:
        zs = points[:, 2]
        return [np.mean(xs), np.mean(ys), np.mean(zs)]
    else:
        return [np.mean(xs), np.mean(ys)]


def _nose_2eyes(landmarks):
    """
    :param landmarks:
    :return: nose, right eye corner, left eye corner
    """
    return np.array([_centroid_nose(landmarks), _centroid_right_eye(landmarks),
                     _centroid_left_eye(landmarks)], dtype=np.double)


def get_model_3d_points(mode='nose_chin_eyes_mouth'):
    """
    :param mode: must in ['nose_eyes_ears','nose_chin_eyes_mouth', 'nose_eyes_mouth', 'nose_2eyes']
    :return:
    """
    model_points = get_full_model_points()
    if mode == 'nose_chin_eyes_mouth':
        return _nose_chin_2eyes_2mouth(model_points)
    elif mode == 'nose_eyes_ears':
        return _nose_2eyes_2ears(model_points)
    elif mode == 'nose_eyes_mouth':
        return _nose_2eyes_2mouth(model_points)
    elif mode == 'nose_2eyes':
        return _nose_2eyes(landmarks=model_points)
    else:
        raise ValueError(
            'mode must be either "nose_eyes_ears" or "nose_chin_eyes_mouth" or "nose_eyes_mouth" or "nose_2eyes"')


def get_points_from_landmarks(landmarks, mode='nose_chin_eyes_mouth'):
    """
    :param landmarks: 68 points of face landmark
    :param mode: must in ['nose_eyes_ears','nose_chin_eyes_mouth', 'nose_eyes_mouth','nose_2eyes']
    :return:
    """
    if mode == 'nose_chin_eyes_mouth':
        return _nose_chin_2eyes_2mouth(landmarks)
    elif mode == 'nose_eyes_ears':
        return _nose_2eyes_2ears(landmarks)
    elif mode == 'nose_eyes_mouth':
        return _nose_2eyes_2mouth(landmarks)
    elif mode == 'nose_2eyes':
        return _nose_2eyes(landmarks)
    else:
        raise ValueError(
            'mode must be either "nose_eyes_ears" or "nose_chin_eyes_mouth" or "nose_eyes_mouth" or "nose_2eyes"')
