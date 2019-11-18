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
