import cv2

from head_pose.face import face_68_landmarks, face_5_landmarks
from sample_images import sample_image


def test():
    image_file = sample_image('yoga_01.jpg')
    im = cv2.imread(image_file)
    landmarks = face_68_landmarks(im)
    for face in landmarks:
        for (x, y) in face:
            cv2.circle(im, (x, y), 2, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.imshow('im', im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def test_5_points_landmarks():
    image_file = sample_image('/Users/administrator/Documents/video/huatuo/math_2018-04-25/images/0356.jpg')
    im = cv2.imread(image_file)
    landmarks = face_5_landmarks(im)
    # print(landmarks[0].shape)
    # print(len(landmarks))
    # print(landmarks[0])
    # print(landmarks[0].part(0))
    # print(landmarks[0].part(1))
    # print(landmarks[0].part(2))
    # print(landmarks[0].part(3))
    # print(landmarks[0].part(4))
    # print(landmarks[0].part(5))

    for face in landmarks:
        print('+++++++++++++++: {}'.format(face))
        print('+++++++++++++++: {}'.format(face.num_parts))
        print('+++++++++++++++: {}'.format(face.part(0)))
        for i in range(face.num_parts):
            point = face.part(i)
            print(type(point))
            cv2.circle(im, (point.x, point.y), 2, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.imshow('im', im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


test_5_points_landmarks()
