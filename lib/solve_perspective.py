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
from scipy import linalg
import logging


def factorize(matrix):
    # QR factorize the submatrix
    r, q = linalg.rq(matrix[:, :3])
    # compute the translation
    t = linalg.lstsq(r, matrix[:, 3:4])[0]

    # fix the intrinsic and rotation matrix
    # intrinsic matrix's diagonal entries must be all positive
    # rotation matrix's determinant must be 1; otherwise there's an reflection component
    #logging.info('before fixing, diag of r: {}, {}, {}'.format(r[0, 0], r[1, 1], r[2, 2]))

    neg_sign_cnt = int(r[0, 0] < 0) + int(r[1, 1] < 0) + int(r[2, 2] < 0)
    if neg_sign_cnt == 1 or neg_sign_cnt == 3:
        r = -r

    new_neg_sign_cnt = int(r[0, 0] < 0) + int(r[1, 1] < 0) + int(r[2, 2] < 0)
    assert (new_neg_sign_cnt == 0 or new_neg_sign_cnt == 2)

    fix = np.diag((1, 1, 1))
    if r[0, 0] < 0 and r[1, 1] < 0:
        fix = np.diag((-1, -1, 1))
    elif r[0, 0] < 0 and r[2, 2] < 0:
        fix = np.diag((-1, 1, -1))
    elif r[1, 1] < 0 and r[2, 2] < 0:
        fix = np.diag((1, -1, -1))
    r = np.dot(r, fix)
    q = np.dot(fix, q)
    t = np.dot(fix, t)

    assert (linalg.det(q) > 0)
    #logging.info('after fixing, diag of r: {}, {}, {}'.format(r[0, 0], r[1, 1], r[2, 2]))

    # check correctness
    # ratio = np.dot(r, np.hstack((q, t))) / matrix
    # assert (np.all(ratio > 0) or np.all(ratio < 0))
    # tmp = np.max(np.abs(np.abs(ratio) - np.ones((3, 4))))
    # logging.info('factorization, max relative error: {}'.format(tmp))
    # assert (np.max(tmp) < 1e-9)

    # normalize the r matrix
    r /= r[2, 2]

    return r, q, t


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

    # with open('/data2/ex1_perspective.txt', 'w') as fp:
    #     fp.write('K:\n{}\n\n'.format(r))
    #     fp.write('R:\n{}\n\n'.format(q))
    #     fp.write('t:\n{}\n'.format(t))

    return r, q, t

