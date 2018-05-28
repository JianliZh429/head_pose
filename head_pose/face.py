# -*- coding: utf-8 -*-
import dlib
import numpy as np

from . import utils

face_detector = dlib.cnn_face_detection_model_v1(
    "/Users/administrator/workspace/AI_models/dlib/mmod_human_face_detector.dat")

face_shape_predictor = dlib.shape_predictor(
    "/Users/administrator/workspace/AI_models/dlib/shape_predictor_5_face_landmarks.dat")

face_descriptor = dlib.face_recognition_model_v1(
    "/Users/administrator/workspace/AI_models/dlib/dlib_face_recognition_resnet_model_v1.dat")

pose_predictor_68_point = dlib.shape_predictor(
    "/Users/administrator/workspace/AI_models/dlib/shape_predictor_68_face_landmarks.dat")

front_face_detector = dlib.get_frontal_face_detector()


def detect_faces(im, cnn=True):
    if cnn:
        faces = face_detector(im)
        return utils.to_dlib_rectangles(faces)
    else:
        return front_face_detector(im, 1)


def compute_face_descriptor(im, face):
    shape = face_shape_predictor(im, face)
    descriptor = face_descriptor.compute_face_descriptor(im, shape, 100)
    return np.array(descriptor)


def _raw_face_landmarks(face_image, cnn=True):
    face_locations = detect_faces(face_image, cnn=cnn)

    return [pose_predictor_68_point(face_image, face_location) for face_location in face_locations]


def face_68_landmarks(face_image, cnn=True):
    """
    Given an image, returns a dict of face feature locations (eyes, nose, etc) for each face in the image
    :param cnn: use cnn to do face detection or hog
    :param face_image: image to search
    :return: A list of dicts of face feature locations (eyes, nose, etc)
    """
    landmarks = _raw_face_landmarks(face_image, cnn=cnn)
    landmarks_as_tuples = [[(p.x, p.y) for p in landmark.parts()] for landmark in landmarks]

    return landmarks_as_tuples


def face_5_landmarks(face_image, cnn=True):
    face_locations = detect_faces(face_image, cnn=cnn)

    # faces = dlib.full_object_detections()
    faces = []
    for face in face_locations:
        faces.append(face_shape_predictor(face_image, face))

    return faces
