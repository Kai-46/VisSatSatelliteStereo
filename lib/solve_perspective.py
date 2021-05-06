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
from scipy import linalg
import logging


def factorize(P):
    P = P[:3, :4]

    # RQ factorize the submatrix
    K, R = linalg.rq(P[:3, :3])

    # fix the intrinsic and rotation matrix
    ##### intrinsic matrix's diagonal entries must be all positive
    ##### rotation matrix's determinant must be 1; otherwise there's an reflection component
    neg_sign_cnt = int(K[0, 0] < 0) + int(K[1, 1] < 0) + int(K[2, 2] < 0)
    if neg_sign_cnt == 1 or neg_sign_cnt == 3:
        K = -K
        R = -R

    new_neg_sign_cnt = int(K[0, 0] < 0) + int(K[1, 1] < 0) + int(K[2, 2] < 0)
    assert (new_neg_sign_cnt == 0 or new_neg_sign_cnt == 2)

    fix = np.diag((1, 1, 1))
    if K[0, 0] < 0 and K[1, 1] < 0:
        fix = np.diag((-1, -1, 1))
    elif K[0, 0] < 0 and K[2, 2] < 0:
        fix = np.diag((-1, 1, -1))
    elif K[1, 1] < 0 and K[2, 2] < 0:
        fix = np.diag((1, -1, -1))
    K = K @ fix
    R = fix @ R

    # normalize the K matrix
    scale = K[2, 2]
    K /= scale
    P[:3, :4] /= scale
    t = linalg.lstsq(K, P[:3, 3:4])[0]

    assert(np.isclose(np.linalg.det(R), 1.))
    assert(np.all(np.isclose(P[:3, :4], K@np.hstack((R, t)))))
    return K, R, t

# colmap convention for pixel indices: (col, row)
def solve_perspective(xx, yy, zz, col, row, keep_mask=None):
    diff_size = np.array([yy.size - xx.size, zz.size - xx.size, col.size - xx.size, row.size - xx.size])
    assert (np.all(diff_size == 0))

    if keep_mask is not None:
        # logging.info('discarding {} % outliers'.format((1. - np.sum(keep_mask) / keep_mask.size) * 100.))
        xx = xx[keep_mask].reshape((-1, 1))
        yy = yy[keep_mask].reshape((-1, 1))
        zz = zz[keep_mask].reshape((-1, 1))
        row = row[keep_mask].reshape((-1, 1))
        col = col[keep_mask].reshape((-1, 1))

    # logging.info('solving perspective, xx: [{}, {}], yy: [{}, {}], zz: [{}, {}]'.format(np.min(xx), np.max(xx),
    #                                                                                     np.min(yy), np.max(yy),
    #                                                                                     np.min(zz), np.max(zz)))
    #
    # scatter3d(xx, yy, zz, '/data2/temp.jpg')

    point_cnt = xx.size
    all_ones = np.ones((point_cnt, 1))
    all_zeros = np.zeros((point_cnt, 4))
    # construct the least square problem
    A1 = np.hstack((xx, yy, zz, all_ones,
                    all_zeros,
                    -col * xx, -col * yy, -col * zz, -col * all_ones))
    A2 = np.hstack((all_zeros,
                    xx, yy, zz, all_ones,
                    -row * xx, -row * yy, -row * zz, -row * all_ones))

    A = np.vstack((A1, A2))
    u, s, vh = linalg.svd(A, full_matrices=False)

    # logging.info('smallest singular value: {}'.format(s[11]))

    P = np.real(vh[11, :]).reshape((3, 4))

    singular_values = ''
    for i in range(11, -1, -1):
        singular_values += ' {}'.format(np.real(s[i]))
    logging.info('singular values: {}'.format(singular_values))

    # factorize into standard form
    r, q, t = factorize(P)

    return r, q, t

