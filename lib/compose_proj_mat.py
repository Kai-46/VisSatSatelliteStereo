# ===============================================================================================================
#  This file is part of Creation of Operationally Realistic 3D Environment (CORE3D).                            =
#  Copyright 2019 Cornell University - All Rights Reserved                                                      =
#  -                                                                                                            =
#  NOTICE: All information contained herein is, and remains the property of General Electric Company            =
#  and its suppliers, if any. The intellectual and technical concepts contained herein are proprietary          =
#  to General Electric Company and its suppliers and may be covered by U.S. and Foreign Patents, patents        =
#  in process, and are protected by trade secret or copyright law. Dissemination of this information or         =
#  reproduction of this material is strictly forbidden unless prior written permission is obtained              =
#  from General Electric Company.                                                                               =
#  -                                                                                                            =
#  The research is based upon work supported by the Office of the Director of National Intelligence (ODNI),     =
#  Intelligence Advanced Research Projects Activity (IARPA), via DOI/IBC Contract Number D17PC00287.            =
#  The U.S. Government is authorized to reproduce and distribute copies of this work for Governmental purposes. =
# ===============================================================================================================

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
