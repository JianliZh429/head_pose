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
        focal_length = image_size[1]
        camera_center = (image_size[1] / 2, image_size[0] / 2)
        self.camera_matrix = np.array([
            [focal_length, 0, camera_center[0]],
            [0, focal_length, camera_center[1]],
            [0, 0, 1]
        ], dtype=np.double)

        self.dist_coeffs = np.zeros((4, 1))

    def solve_pose(self, image_points):
        """
        Solve pose from image points, if length is 3, order is sensitive, for example: nose, right eye, left eye
        Return (rotation_vector, translation_vector) as pose.
        """
        if len(image_points) == 3:
            nose = image_points[0]
            right = image_points[1]
            left = image_points[2]
            dist1 = np.linalg.norm(nose - right)
            dist2 = np.linalg.norm(nose - left)
            if dist1 <= dist2:
                r_vec = np.array([[0.34554543], [-0.72173726], [0.08495318]])
                t_vec = np.array([[-12.14525577], [-48.03475936], [383.82047981]])
            else:
                r_vec = np.array([[0.75807009], [0.3207348], [-2.80691676]])
                t_vec = np.array([[-24.07046963], [-1.68285571], [-199.17583135]])
            # else:
            #     r_vec = np.array([[0.0], [-0.0], [0.0]])
            #     t_vec = np.array([[[0.0], [0.0], [0.0]]])
            (_, rotation_vector, translation_vector) = cv2.solvePnP(self.model_points_3d, image_points,
                                                                    self.camera_matrix, self.dist_coeffs,
                                                                    rvec=r_vec, tvec=t_vec,
                                                                    useExtrinsicGuess=True,
                                                                    flags=cv2.SOLVEPNP_ITERATIVE)
        else:
            (_, rotation_vector, translation_vector) = cv2.solvePnP(self.model_points_3d, image_points,
                                                                    self.camera_matrix, self.dist_coeffs)

            # (_, rotation_vector, translation_vector) = cv2.solvePnP(self.model_points_3d, image_points,
            #                                                         self.camera_matrix, self.dist_coeffs,
            #                                                         rvec=rotation_vector,
            #                                                         tvec=translation_vector,
            #                                                         useExtrinsicGuess=True,
            #                                                         flags=cv2.SOLVEPNP_ITERATIVE)
        return rotation_vector, translation_vector

    def projection(self, rotation_vector, translation_vector, cube_edge=50.0):
        """
        :param rotation_vector:
        :param translation_vector:
        :param cube_edge: the length of cube edge
        :return:
        """
        points_3d = [(cube_edge, cube_edge, cube_edge), (cube_edge, cube_edge, -cube_edge),
                     (cube_edge, -cube_edge, -cube_edge), (cube_edge, -cube_edge, cube_edge),
                     (-cube_edge, cube_edge, cube_edge), (-cube_edge, cube_edge, -cube_edge),
                     (-cube_edge, -cube_edge, -cube_edge), (-cube_edge, -cube_edge, cube_edge)]

        points_3d = np.array(points_3d, dtype=np.float).reshape(-1, 3)
        (points_2d, _) = cv2.projectPoints(points_3d, rotation_vector, translation_vector,
                                           self.camera_matrix, self.dist_coeffs)
        return points_2d.reshape(-1, 2)
