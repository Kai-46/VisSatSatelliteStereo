import numpy as np
import logging

def solve_affine(xx, yy, zz, col, row, keep_mask=None):
    diff_size = np.array([yy.size - xx.size, zz.size - xx.size, col.size - xx.size, row.size - xx.size])
    assert (np.all(diff_size == 0))

    if keep_mask is not None:
        logging.info('discarding {} % outliers'.format((1. - np.sum(keep_mask) / keep_mask.size) * 100.))
        xx = xx[keep_mask].reshape((-1, 1))
        yy = yy[keep_mask].reshape((-1, 1))
        zz = zz[keep_mask].reshape((-1, 1))
        row = row[keep_mask].reshape((-1, 1))
        col = col[keep_mask].reshape((-1, 1))

    # construct a least square problem
    logging.info('\n\n')
    logging.info('xx: {}, {}'.format(np.min(xx), np.max(xx)))
    logging.info('yy: {}, {}'.format(np.min(yy), np.max(yy)))
    logging.info('zz: {}, {}'.format(np.min(zz), np.max(zz)))
    logging.info('col: {}, {}'.format(np.min(col), np.max(col)))
    logging.info('row: {}, {}'.format(np.min(row), np.max(row)))

    point_cnt = xx.size
    all_ones = np.ones((point_cnt, 1))
    all_zeros = np.zeros((point_cnt, 4))
    # construct the least square problem
    A1 = np.hstack((xx, yy, zz, all_ones, all_zeros))
    A2 = np.hstack((all_zeros, xx, yy, zz, all_ones))

    A = np.vstack((A1, A2))
    b = np.vstack((col, row))
    res = np.linalg.lstsq(A, b)

    logging.info('residual error (pixels): {}'.format(np.sqrt(res[1][0] / point_cnt)))

    P = res[0].reshape((2, 4))

    return P