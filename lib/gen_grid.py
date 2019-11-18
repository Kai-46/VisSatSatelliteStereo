#  ===============================================================================================================
#  Copyright (c) 2019, Cornell University. All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without modification, are permitted provided that
#  the following conditions are met:
#
#      * Redistributions of source code must retain the above copyright otice, this list of conditions and
#        the following disclaimer.
#
#      * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and
#        the following disclaimer in the documentation and/or other materials provided with the distribution.
#
#      * Neither the name of Cornell University nor the names of its contributors may be used to endorse or
#        promote products derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
#  WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
#  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE
#  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
#  TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
#  HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
#   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
#  OF SUCH DAMAGE.
#
#  Author: Kai Zhang (kz298@cornell.edu)
#
#  The research is based upon work supported by the Office of the Director of National Intelligence (ODNI),
#  Intelligence Advanced Research Projects Activity (IARPA), via DOI/IBC Contract Number D17PC00287.
#  The U.S. Government is authorized to reproduce and distribute copies of this work for Governmental purposes.
#  ===============================================================================================================


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
