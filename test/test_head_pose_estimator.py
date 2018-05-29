import os

import cv2
import numpy as np

from head_pose import get_points_from_landmarks, HeadPoseEstimator
from head_pose.face import face_68_landmarks
from sample_images import sample_images

BASE_DIR = os.path.dirname(__file__)


def estimate(image_file, mode='nose_2eyes'):
    im = cv2.imread(image_file)
    landmarks = face_68_landmarks(im)
    height, width = im.shape[:2]
    print(height, width)

    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 255, 0), (255, 125, 125)]

    pose_estimator = HeadPoseEstimator(image_size=(height, width), mode=mode)
    for marks in landmarks:
        # for pnt in marks:
        #     cv2.circle(im, (int(pnt[0]), int(pnt[1])), 1, (0, 255, 0), 2, cv2.LINE_AA)
        image_points = get_points_from_landmarks(marks, mode)
        print('========len======== : ', len(image_points))
        print('========   ======== : ', image_points)
        rotation_vector, translation_vector = pose_estimator.solve_pose(image_points)
        print('-------------------------\n', rotation_vector, '||||\n', translation_vector,
              '\n-------------------------\n')
        end_points_2d = pose_estimator.projection(rotation_vector, translation_vector)

        for i, pnt in enumerate(image_points.tolist()):
            cv2.circle(im, (int(pnt[0]), int(pnt[1])), 1, colors[i % 6], 3, cv2.LINE_AA)

        end_points_2d = np.array(end_points_2d).astype(np.int).tolist()
        cv2.line(im, tuple(end_points_2d[5]), tuple(end_points_2d[6]), (0, 255, 0))
        cv2.line(im, tuple(end_points_2d[6]), tuple(end_points_2d[7]), (255, 0, 0))
        cv2.line(im, tuple(end_points_2d[2]), tuple(end_points_2d[6]), (0, 0, 255))
    return im
    # cv2.imshow('im', im)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


OUTPUT = 'output_nose_2eyes_02'


def test(mode='nose_eyes_mouth'):
    image_files = sample_images(os.path.join(BASE_DIR, '../sample_images'))
    print(image_files)
    output_dir = os.path.join(BASE_DIR, '../{}'.format(OUTPUT))
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    for im_f in image_files:
        f_name = im_f.split(os.sep)[-1]
        print('\n------------------------{}'.format(f_name))
        im = estimate(im_f, mode)
        cv2.imwrite('{}/{}'.format(output_dir, f_name), im)


# test('nose_eyes_ears')
# test('nose_chin_eyes_mouth')
# test('nose_eyes_mouth')
test('nose_2eyes')
