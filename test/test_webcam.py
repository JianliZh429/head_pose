import cv2
import numpy as np

from head_pose import HeadPoseEstimator, get_points_from_landmarks
from head_pose.face import face_68_landmarks


def pose_estimate():
    landmarks = face_68_landmarks(frame, cnn=False)
    height, width = frame.shape[:2]
    print(height, width)

    pose_estimator = HeadPoseEstimator(image_size=(height, width), mode='nose_eyes_mouth')
    for marks in landmarks:
        image_points = get_points_from_landmarks(marks, mode='nose_eyes_mouth')
        rotation_vector, translation_vector = pose_estimator.solve_pose(image_points)
        print('-----------------------------------')
        print(rotation_vector, '===\n', translation_vector)
        nose_end_points_2d = pose_estimator.projection(rotation_vector, translation_vector)

        for pnt in image_points.tolist():
            cv2.circle(frame, (int(pnt[0]), int(pnt[1])), 1, (0, 255, 0), 1, cv2.LINE_AA)
        for pnt in nose_end_points_2d.tolist():
            cv2.circle(frame, (int(pnt[0]), int(pnt[1])), 1, (0, 0, 255), 1, cv2.LINE_AA)
        nose_end_points_2d = np.array(nose_end_points_2d).astype(np.int).tolist()
        cv2.line(frame, tuple(nose_end_points_2d[5]), tuple(nose_end_points_2d[6]), (0, 255, 0))
        cv2.line(frame, tuple(nose_end_points_2d[6]), tuple(nose_end_points_2d[7]), (255, 0, 0))
        cv2.line(frame, tuple(nose_end_points_2d[2]), tuple(nose_end_points_2d[6]), (0, 0, 255))


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        pose_estimate()
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
