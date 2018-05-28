import cv2
import numpy as np

from .model_loader import get_model_3d_points


class HeadPoseEstimator:
    def __init__(self, image_size=(480, 640), mode='nose_chin_eyes_mouth'):
        """
        :param image_size:
        :param mode: must in ['nose_eyes_ears','nose_chin_eyes_mouth', 'nose_eyes_mouth','nose_2eyes']
        """
        self.model_points_3d = get_model_3d_points(mode=mode)
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
        if len(image_points) == 3:

            # r_vec = np.array([[0.31129965], [-0.61099538], [2.76988726]])
            # t_vec = np.array([[22.16439846], [-13.85181104], [-303.74858751]])
            # r_vec = np.array([[0.67449138], [-0.06618537], [-3.07730173]])
            # t_vec = np.array([[-131.09678991], [-66.22593863], [-1075.76498503]])
            r_vec = np.array([[0.01891013], [0.08560084], [-3.14392813]])
            t_vec = np.array([[-14.97821226], [-10.62040383], [-2053.03596872]])
            (_, rotation_vector, translation_vector) = cv2.solvePnP(self.model_points_3d, image_points,
                                                                    self.camera_matrix, self.dist_coeffs,
                                                                    rvec=r_vec, tvec=t_vec,
                                                                    useExtrinsicGuess=True,
                                                                    flags=cv2.SOLVEPNP_ITERATIVE)
        else:
            (_, rotation_vector, translation_vector) = cv2.solvePnP(self.model_points_3d, image_points,
                                                                    self.camera_matrix, self.dist_coeffs)
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
