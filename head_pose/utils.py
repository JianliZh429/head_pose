# -*- coding: utf-8 -*-
import math

import cv2
import dlib
import numpy as np


def to_dlib_rectangles(mmod_rectangles):
    rectangles = dlib.rectangles()
    rectangles.extend([d.rect for d in mmod_rectangles])
    return rectangles


def to_rectangle(dlib_rectangle):
    return dlib_rectangle.left(), dlib_rectangle.top(), dlib_rectangle.right(), dlib_rectangle.bottom()


def to_pitch_yaw_roll(rotation_vector, translation_vector):
    rotation_matrix = cv2.Rodrigues(rotation_vector)[0]
    projection_matrix = np.hstack((rotation_matrix, translation_vector))
    euler_angles = cv2.decomposeProjectionMatrix(projection_matrix)[6]

    pitch, yaw, roll = [math.radians(e) for e in euler_angles.squeeze()]
    
    pitch = math.degrees(math.asin(math.sin(pitch)))
    roll = math.degrees(math.asin(math.sin(roll)))
    yaw = -math.degrees(math.asin(math.sin(yaw)))
    return pitch, yaw, roll
