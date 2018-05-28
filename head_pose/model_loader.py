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


def _nose_2eyes_2mouth(landmarks):
    """
    :param landmarks:
    :return: nose, right eye corner, left eye corner, right mouth corner, left mouth corner
    """
    return np.array([landmarks[30], landmarks[36], landmarks[45], landmarks[48], landmarks[54]], dtype=np.double)


def _centroid_nose(landmarks):
    noses = np.array([landmarks[30], landmarks[31], landmarks[32], landmarks[33], landmarks[34], landmarks[35]])
    return _centroid(noses)


def _centroid_right_eye(landmarks):
    right_eyes = np.array([landmarks[36], landmarks[37], landmarks[38], landmarks[39], landmarks[40], landmarks[41]])
    return _centroid(right_eyes)


def _centroid_left_eye(landmarks):
    left_eyes = np.array([landmarks[42], landmarks[43], landmarks[44], landmarks[45], landmarks[46], landmarks[47]])
    return _centroid(left_eyes)


def _centroid(points):
    print('----------------')
    print(len(points))
    print(points.shape)
    print(points)
    xs = points[:, 0]
    ys = points[:, 1]

    if len(points[0]) == 3:
        zs = points[:, 2]
        print('xs: ', xs)
        print(np.mean(xs))
        return [np.mean(xs), np.mean(ys), np.mean(zs)]
    else:
        return [np.mean(xs), np.mean(ys)]


def _nose_2eyes(landmarks):
    """
    :param landmarks:
    :return: nose, right eye corner, left eye corner
    """
    _68_face_marks = landmarks
    return np.array([_centroid_nose(_68_face_marks), _centroid_right_eye(_68_face_marks),
                     _centroid_left_eye(_68_face_marks)], dtype=np.double)


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
