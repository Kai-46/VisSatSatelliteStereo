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

# point_cnt = x_point_cnt * y_point_cnt * z_point_cnt
#
# x_points = np.linspace(x_min, x_max, x_point_cnt)
# y_points = np.linspace(y_min, y_max, y_point_cnt)
# z_points = np.linspace(z_min, z_max, z_point_cnt)

# generate a 3D grid
# x_points, y_points, z_points are numpy array
def gen_grid(x_points, y_points, z_points):
    x_point_cnt = x_points.size
    y_point_cnt = y_points.size
    z_point_cnt = z_points.size
    point_cnt = x_point_cnt * y_point_cnt * z_point_cnt

    xx, yy = np.meshgrid(x_points, y_points, indexing='ij')
    xx = np.reshape(xx, (-1, 1))
    yy = np.reshape(yy, (-1, 1))
    xx = np.tile(xx, (z_point_cnt, 1))
    yy = np.tile(yy, (z_point_cnt, 1))

    zz = np.zeros((point_cnt, 1))
    for j in range(z_point_cnt):
        idx1 = j * x_point_cnt * y_point_cnt
        idx2 = (j + 1) * x_point_cnt * y_point_cnt
        zz[idx1:idx2, 0] = z_points[j]

    return xx, yy, zz
