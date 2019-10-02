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
import logging


def esti_linear(source, target):
    samples_cnt = source.shape[0]

    # for numeric stability
    source_mean = np.mean(source, axis=0).reshape((1, 3))
    target_mean = np.mean(target, axis=0).reshape((1, 3))

    source_tmp = source - np.tile(source_mean, (samples_cnt, 1))
    target_tmp = target - np.tile(target_mean, (samples_cnt, 1))

    all_zeros = np.zeros((samples_cnt, 1))
    all_ones = np.ones((samples_cnt, 1))

    A1 = np.hstack((source_tmp, np.tile(all_zeros, (1, 6)), all_ones, np.tile(all_zeros, (1, 2))))
    A2 = np.hstack((np.tile(all_zeros, (1, 3)), source_tmp, np.tile(all_zeros, (1, 3)), all_zeros, all_ones, all_zeros))
    A3 = np.hstack((np.tile(all_zeros, (1, 6)), source_tmp, np.tile(all_zeros, (1, 2)), all_ones))

    A = np.vstack((A1, A2, A3))
    b = np.vstack((target_tmp[:, 0:1], target_tmp[:, 1:2], target_tmp[:, 2:3]))

    result = np.linalg.lstsq(A, b, rcond=-1)[0]
    M = result[0:9].reshape((3, 3))
    t = result[9:12].reshape((3, 1))

    # add subtracted mean
    t = -np.dot(M, source_mean.T) + t + target_mean.T

    M = M.T
    t = t.T

    logging.info('\nestimated: ')
    logging.info('M: {}'.format(M))
    logging.info('t: {}'.format(t))

    # check whether M is close to a rotation matrix
    u, s, vh = np.linalg.svd(M)
    cond = s[0] / s[2]
    logging.info('condition number of M: {}, smallest singular value: {}'.format(cond, s[2]))
    # assert(s[2] > 0 and cond < 1.5)

    source_aligned = np.dot(source, M) + np.tile(t, (samples_cnt, 1))

    # compute re-projection error over the whole set
    err = np.sqrt(np.sum((source - target) ** 2, axis=1))
    logging.info('alignment error before, min: {}, max: {}, mean: {}, median: {}'.format(np.min(err), np.max(err), np.mean(err), np.median(err)))
    err = np.sqrt(np.sum((source_aligned - target) ** 2, axis=1))
    logging.info('alignment error after, min: {}, max: {}, mean: {}, median: {}'.format(np.min(err), np.max(err), np.mean(err), np.median(err)))

    return M, t