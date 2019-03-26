import numpy as np
from pyquaternion import Quaternion


def compose_proj_mat(params, cam_type='perspective'):
    if cam_type == 'perspective':
        # fx, fy, cx, cy, s, qvec, t
        fx = params[0]
        fy = params[1]
        cx = params[2]
        cy = params[3]
        s = params[4]
        qvec = params[5:9]
        tvec = params[9:12]
    else:
        # fx, fy, cx, cy, qvec, t
        fx = params[0]
        fy = params[1]
        cx = params[2]
        cy = params[3]
        s = 0.
        qvec = params[4:8]
        tvec = params[8:11]

    K = np.array([[fx, s, cx],
                  [0., fy, cy],
                  [0., 0., 1.]])
    R = Quaternion(qvec[0], qvec[1], qvec[2], qvec[3]).rotation_matrix
    tvec = np.reshape(tvec, (-1, 1))
    P = np.dot(K, np.hstack((R, tvec)))
    return P
