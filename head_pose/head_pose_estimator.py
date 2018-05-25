import cv2
import numpy as np

from head_pose.model_loader import get_nose_eye_chin_mouth_6


class HeadPoseEstimator:
    def __init__(self, image_size=(480, 640)):
        self.model_points_3d = get_nose_eye_chin_mouth_6()
        print('-----: {}'.format(self.model_points_3d))
        focal_length = image_size[1]
        camera_center = (image_size[1] / 2, image_size[0] / 2)
        self.camera_matrix = np.array([
            [focal_length, 0, camera_center[0]],
            [0, focal_length, camera_center[1]],
            [0, 0, 1]
        ], dtype="double")

        self.dist_coeffs = np.zeros((4, 1))

    def solve_pose(self, image_points):
        """
        Solve pose from image points
        Return (rotation_vector, translation_vector) as pose.
        """
        (_, rotation_vector, translation_vector) = cv2.solvePnP(self.model_points_3d, image_points,
                                                                self.camera_matrix, self.dist_coeffs,
                                                                flags=cv2.SOLVEPNP_ITERATIVE)

        # (success, rotation_vector, translation_vector) = cv2.solvePnP(
        #     self.model_points,
        #     image_points,
        #     self.camera_matrix,
        #     self.dist_coeffs,
        #     rvec=self.r_vec,
        #     tvec=self.t_vec,
        #     useExtrinsicGuess=True)
        return rotation_vector, translation_vector

    def projection(self, rotation_vector, translation_vector):
        points_3d = [(50.0, 50.0, 50.0), (50.0, 50.0, -50.0), (50.0, -50.0, -50.0), (50.0, -50.0, 50.0),
                     (-50.0, 50.0, 50.0), (-50.0, 50.0, -50.0), (-50.0, -50.0, -50.0), (-50.0, -50.0, 50.0)]
        # points_3d = [(0, -50, 0), (50.0, -50, 0), (0, 0, 0), (0, -50, 50.0)]
        points_3d = np.array(points_3d, dtype=np.float).reshape(-1, 3)
        (points_2d, _) = cv2.projectPoints(points_3d, rotation_vector, translation_vector,
                                           self.camera_matrix, self.dist_coeffs)
        return points_2d.reshape(-1, 2)
