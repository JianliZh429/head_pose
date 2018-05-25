import cv2
import numpy as np

from head_pose import face_68_landmarks, get_nose_eye_chin_mouth_2d, HeadPoseEstimator
from sample_images import sample_image


def test():
    image_file = sample_image('danielle-bregoli.jpg')
    im = cv2.imread(image_file)
    landmarks = face_68_landmarks(im)
    height, width = im.shape[:2]
    print(height, width)

    pose_estimator = HeadPoseEstimator(image_size=(height, width))
    for marks in landmarks:
        image_points = get_nose_eye_chin_mouth_2d(marks)
        print('========len======== : ', len(image_points))
        print('========   ========: ', image_points)
        rotation_vector, translation_vector = pose_estimator.solve_pose(image_points)
        print(rotation_vector, translation_vector)
        end_points_2d = pose_estimator.projection(rotation_vector, translation_vector)

        for pnt in end_points_2d.tolist():
            cv2.circle(im, (int(pnt[0]), int(pnt[1])), 1, (0, 0, 255), 1, cv2.LINE_AA)
        end_points_2d = np.array(end_points_2d).astype(np.int).tolist()
        cv2.line(im, tuple(end_points_2d[5]), tuple(end_points_2d[6]), (0, 255, 0))
        cv2.line(im, tuple(end_points_2d[6]), tuple(end_points_2d[7]), (255, 0, 0))
        cv2.line(im, tuple(end_points_2d[2]), tuple(end_points_2d[6]), (0, 0, 255))
    cv2.imshow('im', im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


test()
