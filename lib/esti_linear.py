import numpy as np
import logging


def esti_linear(source, target):
    samples_cnt = source.shape[0]

    all_zeros = np.zeros((samples_cnt, 1))
    all_ones = np.ones((samples_cnt, 1))

    A1 = np.hstack((source, np.tile(all_zeros, (1, 6)), all_ones, np.tile(all_zeros, (1, 2))))
    A2 = np.hstack((np.tile(all_zeros, (1, 3)), source, np.tile(all_zeros, (1, 3)), all_zeros, all_ones, all_zeros))
    A3 = np.hstack((np.tile(all_zeros, (1, 6)), source, np.tile(all_zeros, (1, 2)), all_ones))

    A = np.vstack((A1, A2, A3))
    b = np.vstack((target[:, 0:1], target[:, 1:2], target[:, 2:3]))

    result = np.linalg.lstsq(A, b, rcond=None)[0]
    M = result[0:9].reshape((3, 3))
    t = result[9:12].reshape((3, 1))

    # check whether M is close to a rotation matrix
    u, s, vh = np.linalg.svd(M)
    # print(M)
    # print('singular values of M: {}'.format(s))
    cond = s[0] / s[2]
    logging.info('condition number of M: {}, smallest singular value: {}'.format(cond, s[2]))
    # assert(s[2] > 0 and cond < 1.5)

    # check the MSE after applying the linear map
    # points_ba_reg = np.dot(points_ba_arr, M.T) + np.tile(t.T, (sample_cnt, 1))
    # mse_before = np.mean((points_arr - points_ba_arr)**2)
    # mse_after = np.mean((points_arr - points_ba_reg)**2)
    # print('mse before: {}'.format(mse_before))
    # print('mse after: {}'.format(mse_after))

    M = M.T
    t = t.T

    source_aligned = np.dot(source, M) + np.tile(t, (samples_cnt, 1))

    # compute re-projection error over the whole set
    err = np.sqrt(np.sum((source - target) ** 2, axis=1))
    logging.info('\talignment error before, min: {}, max: {}, mean: {}, median: {}'.format(np.min(err), np.max(err), np.mean(err), np.median(err)))
    err = np.sqrt(np.sum((source_aligned - target) ** 2, axis=1))
    logging.info('\talignment error after, min: {}, max: {}, mean: {}, median: {}'.format(np.min(err), np.max(err), np.mean(err), np.median(err)))

    return M, t