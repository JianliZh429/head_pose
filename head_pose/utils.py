# -*- coding: utf-8 -*-
import dlib


def to_dlib_rectangles(mmod_rectangles):
    rectangles = dlib.rectangles()
    rectangles.extend([d.rect for d in mmod_rectangles])
    return rectangles


def to_rectangle(dlib_rectangle):
    return dlib_rectangle.left(), dlib_rectangle.top(), dlib_rectangle.right(), dlib_rectangle.bottom()
