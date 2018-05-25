import cv2

from head_pose import face_68_landmarks
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


test()
