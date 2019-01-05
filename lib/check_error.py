import logging
import numpy as np


def check_perspective_error(xx, yy, zz, col, row, r, q, t, keep_mask):
    if keep_mask is not None:
        # logging.info('discarding {} % outliers'.format((1. - np.sum(keep_mask) / keep_mask.size) * 100.))
        xx = xx[keep_mask].reshape((-1, 1))
        yy = yy[keep_mask].reshape((-1, 1))
        zz = zz[keep_mask].reshape((-1, 1))
        row = row[keep_mask].reshape((-1, 1))
        col = col[keep_mask].reshape((-1, 1))

    point_cnt = xx.size
    all_ones = np.ones((point_cnt, 1))

    # # check the order of the quantities
    # logging.info('\n')
    # logging.info('fx: {}, fy: {}, cx: {} cy: {}, skew: {}'.format(r[0, 0], r[1, 1], r[0, 2], r[1, 2], r[0, 1]))

    translation = np.tile(t.T, (point_cnt, 1))
    result = np.dot(np.hstack((xx, yy, zz)), q.T) + translation
    cam_xx = result[:, 0:1]
    cam_yy = result[:, 1:2]
    cam_zz = result[:, 2:3]

    # logging.info("cam_xx: {}, {}".format(np.min(cam_xx), np.max(cam_xx)))
    # logging.info("cam_yy: {}, {}".format(np.min(cam_yy), np.max(cam_yy)))
    # logging.info("cam_zz: {}, {}".format(np.min(cam_zz), np.max(cam_zz)))

    # # drift caused by skew
    # drift = r[0, 1] * cam_yy / cam_zz
    # min_drift = np.min(drift)
    # max_drift = np.max(drift)
    # logging.info("drift caused by skew (pixel): {}, {}, {}".format(min_drift, max_drift, max_drift - min_drift))

    # decompose the intrinsic
    # logging.info("normalized skew: {}, fx: {}, fy: {}, cx: {}, cy: {}".format(
    #     r[0, 1] / r[1, 1], r[0, 0], r[1, 1], r[0, 2] - r[0, 1] * r[1, 2] / r[1, 1], r[1, 2]))

    # check projection accuracy
    P_hat = np.dot(r, np.hstack((q, t)))
    result = np.dot(np.hstack((xx, yy, zz, all_ones)), P_hat.T)
    esti_col = result[:, 0:1] / result[:, 2:3]
    esti_row = result[:, 1:2] / result[:, 2:3]

    # max_row_err = np.max(np.abs(esti_row - row))
    # max_col_err = np.max(np.abs(esti_col - col))
    # logging.info('projection accuracy, max_row_err: {}, max_col_err: {}'.format(max_row_err, max_col_err))

    # pixel error
    proj_err = np.sqrt((esti_row - row) ** 2 + (esti_col - col) ** 2)
    #logging.info('proj_err (pixels), min, mean, median, max: {}, {}'.format(np.min(proj_err), np.mean(proj_err), np.median(proj_err), np.max(proj_err)))

    # check inverse projection accuracy
    # assume the matching are correct
    result = np.dot(np.hstack((col, row, all_ones)), np.linalg.inv(r.T))
    esti_cam_xx = result[:, 0:1]  # in camera coordinate
    esti_cam_yy = result[:, 1:2]
    esti_cam_zz = result[:, 2:3]

    # compute scale
    scale = (cam_xx * esti_cam_xx + cam_yy * esti_cam_yy + cam_zz * esti_cam_zz) / (
            esti_cam_xx * esti_cam_xx + esti_cam_yy * esti_cam_yy + esti_cam_zz * esti_cam_zz)
    # assert (np.all(scale > 0))
    esti_cam_xx = esti_cam_xx * scale
    esti_cam_yy = esti_cam_yy * scale
    esti_cam_zz = esti_cam_zz * scale

    # check accuarcy in camera coordinate frame
    # logging.info('inverse projection accuracy, cam xx: {}'.format(np.max(np.abs(esti_cam_xx - cam_xx))))
    # logging.info('inverse projection accuracy, cam yy: {}'.format(np.max(np.abs(esti_cam_yy - cam_yy))))
    # logging.info('inverse projection accuracy, cam zz: {}'.format(np.max(np.abs(esti_cam_zz - cam_zz))))

    # inv proj. err
    inv_proj_err = np.sqrt((esti_cam_xx - cam_xx) ** 2 + (esti_cam_yy - cam_yy) ** 2 + (esti_cam_zz - cam_zz) ** 2)
    #logging.info('inv_proj_err (meters), min, mean, median, max: {}, {}'.format(np.min(inv_proj_err), np.mean(inv_proj_err), np.median(inv_proj_err), np.max(inv_proj_err)))

    # check accuracy in object coordinate frame
    # result = np.dot(np.hstack((esti_cam_xx, esti_cam_yy, esti_cam_zz)) - translation, linalg.inv(q.T))
    # esti_xx = result[:, 0:1]
    # esti_yy = result[:, 1:2]
    # esti_zz = result[:, 2:3]

    # logging.info('inverse projection accuracy, xx: {}'.format(np.max(np.abs(esti_xx - xx))))
    # logging.info('inverse projection accuracy, yy: {}'.format(np.max(np.abs(esti_yy - yy))))
    # logging.info('inverse projection accuracy, zz: {}'.format(np.max(np.abs(esti_zz - zz))))

    return np.mean(proj_err), np.median(proj_err), np.max(proj_err), np.mean(inv_proj_err), np.median(inv_proj_err), np.max(inv_proj_err)
