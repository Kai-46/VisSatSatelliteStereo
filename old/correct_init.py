import numpy as np
import json
from lib.esti_linear import esti_linear
from lib.solve_perspective import factorize
from pyquaternion import Quaternion
import logging


# correct linear distortion in the scene
def correct_init(distorted, undistorted, source_camera_dict):
    # estimate linear transformation
    M, b = esti_linear(undistorted, distorted)
    # source, target use row vector
    M = M.T
    b = b.T

    target_camera_dict = {}

    for img_name in sorted(source_camera_dict.keys()):
        # w, h, fx, fy, cx, cy, s, qvec, t
        params = source_camera_dict[img_name]
        w = params[0]
        h = params[1]
        fx = params[2]
        fy = params[3]
        cx = params[4]
        cy = params[5]
        s = params[6]
        qvec = params[7:11]
        tvec = params[11:14]

        K = np.array([[fx, s, cx],
                      [0., fy, cy],
                      [0., 0., 1.]])
        R = Quaternion(qvec[0], qvec[1], qvec[2], qvec[3]).rotation_matrix
        tvec = np.array(tvec).reshape((3, 1))

        K_hat, R_hat, tvec_hat = factorize(np.hstack((np.dot(R, M), np.dot(R, b)+tvec)))

        K_hat = np.dot(K, K_hat)
        # normalize K_hat
        K_hat = K_hat / K_hat[2, 2]

        qvec_hat = Quaternion(matrix=R_hat)

        # w, h, fx, fy, cx, cy, s, qvec, t
        params_hat = [w, h, K_hat[0, 0], K_hat[1, 1], K_hat[0, 2], K_hat[1, 2], K_hat[0, 1],
                                     qvec_hat[0], qvec_hat[1], qvec_hat[2], qvec_hat[3],
                                     tvec_hat[0, 0], tvec_hat[1, 0], tvec_hat[2, 0]]
        target_camera_dict[img_name] = params_hat

        logging.info("old w, h, fx, fy, cx, cy, s, qvec, t: {}".format(params))
        logging.info("updated w, h, fx, fy, cx, cy, s, qvec, t: {}".format(params_hat))

    return target_camera_dict
