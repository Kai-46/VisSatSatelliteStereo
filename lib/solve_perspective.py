import numpy as np
from scipy import linalg


def factorize(matrix):
    # QR factorize the submatrix
    r, q = linalg.rq(matrix[:, :3])
    # compute the translation
    t = linalg.lstsq(r, matrix[:, 3:4])[0]

    # fix the intrinsic and rotation matrix
    # intrinsic matrix's diagonal entries must be all positive
    # rotation matrix's determinant must be 1
    print('before fixing, diag of r: {}, {}, {}'.format(r[0, 0], r[1, 1], r[2, 2]))
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
    print('after fixing, diag of r: {}, {}, {}'.format(r[0, 0], r[1, 1], r[2, 2]))

    # check correctness
    ratio = np.dot(r, np.hstack((q, t))) / matrix
    assert (np.all(ratio > 0) or np.all(ratio < 0))
    tmp = np.max(np.abs(np.abs(ratio) - np.ones((3, 4))))
    print('factorization, max relative error: {}'.format(tmp))
    assert (np.max(tmp) < 1e-9)

    # normalize the r matrix
    r /= r[2, 2]

    return r, q, t


# colmap convention for pixel indices: (col, row)
def solve_perspective(xx, yy, zz, col, row, keep_mask=None):
    diff_size = np.array([yy.size - xx.size, zz.size - xx.size, col.size - xx.size, row.size - xx.size])
    assert (np.all(diff_size == 0))

    if keep_mask is not None:
        print('discarding {} % outliers'.format((1. - np.sum(keep_mask) / keep_mask.size) * 100.))
        xx = xx[keep_mask].reshape((-1, 1))
        yy = yy[keep_mask].reshape((-1, 1))
        zz = zz[keep_mask].reshape((-1, 1))
        row = row[keep_mask].reshape((-1, 1))
        col = col[keep_mask].reshape((-1, 1))

    print('xx: {}, {}'.format(np.min(xx), np.max(xx)))
    print('yy: {}, {}'.format(np.min(yy), np.max(yy)))
    print('zz: {}, {}'.format(np.min(zz), np.max(zz)))
    print('col: {}, {}'.format(np.min(col), np.max(col)))
    print('row: {}, {}'.format(np.min(row), np.max(row)))

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
    u, s, vh = linalg.svd(A)
    print('smallest singular value: {}'.format(s[11]))
    P = np.real(vh[11, :]).reshape((3, 4))

    # factorize into standard form
    r, q, t = factorize(P)

    check_accuracy(xx, yy, zz, col, row, r, q, t)

    return r, q, t

def check_accuracy(xx, yy, zz, col, row, r, q, t):
    point_cnt = xx.size

    # check the order of the quantities
    print('fx: {}, fy: {}, cx: {} cy: {}, skew: {}'.format(r[0, 0], r[1, 1], r[0, 2], r[1, 2], r[0, 1]))
    translation = np.tile(t.T, (point_cnt, 1))
    result = np.dot(np.hstack((xx, yy, zz)), q.T) + translation
    cam_xx = result[:, 0:1]
    cam_yy = result[:, 1:2]
    cam_zz = result[:, 2:3]
    print("cam_xx: {}, {}".format(np.min(cam_xx), np.max(cam_xx)))
    print("cam_yy: {}, {}".format(np.min(cam_yy), np.max(cam_yy)))
    print("cam_zz: {}, {}".format(np.min(cam_zz), np.max(cam_zz)))
    # drift caused by skew
    drift = r[0, 1] * cam_yy / cam_zz
    min_drift = np.min(drift)
    max_drift = np.max(drift)
    print("drift caused by skew (pixel): {}, {}, {}".format(min_drift, max_drift, max_drift - min_drift))

    # decompose the intrinsic
    # print("normalized skew: {}, fx: {}, fy: {}, cx: {}, cy: {}".format(
    #     r[0, 1] / r[1, 1], r[0, 0], r[1, 1], r[0, 2] - r[0, 1] * r[1, 2] / r[1, 1], r[1, 2]))

    # check projection accuracy
    P_hat = np.dot(r, np.hstack((q, t)))
    result = np.dot(np.hstack((xx, yy, zz, all_ones)), P_hat.T)
    esti_col = result[:, 0:1] / result[:, 2:3]
    esti_row = result[:, 1:2] / result[:, 2:3]
    max_row_err = np.max(np.abs(esti_row - row))
    max_col_err = np.max(np.abs(esti_col - col))
    print('projection accuracy, max_row_err: {}, max_col_err: {}'.format(max_row_err, max_col_err))

    # check inverse projection accuracy
    # assume the matching are correct
    result = np.dot(np.hstack((col, row, all_ones)), linalg.inv(r.T))
    esti_cam_xx = result[:, 0:1]  # in camera coordinate
    esti_cam_yy = result[:, 1:2]
    esti_cam_zz = result[:, 2:3]

    # compute scale
    scale = (cam_xx * esti_cam_xx + cam_yy * esti_cam_yy + cam_zz * esti_cam_zz) / (
            esti_cam_xx * esti_cam_xx + esti_cam_yy * esti_cam_yy + esti_cam_zz * esti_cam_zz)
    assert (np.all(scale > 0))
    esti_cam_xx = esti_cam_xx * scale
    esti_cam_yy = esti_cam_yy * scale
    esti_cam_zz = esti_cam_zz * scale

    # check accuarcy in camera coordinate frame
    print('inverse projection accuracy, cam xx: {}'.format(np.max(np.abs(esti_cam_xx - cam_xx))))
    print('inverse projection accuracy, cam yy: {}'.format(np.max(np.abs(esti_cam_yy - cam_yy))))
    print('inverse projection accuracy, cam zz: {}'.format(np.max(np.abs(esti_cam_zz - cam_zz))))
    # check accuracy in object coordinate frame
    result = np.dot(np.hstack((esti_cam_xx, esti_cam_yy, esti_cam_zz)) - translation, linalg.inv(q.T))
    esti_xx = result[:, 0:1]
    esti_yy = result[:, 1:2]
    esti_zz = result[:, 2:3]
    print('inverse projection accuracy, xx: {}'.format(np.max(np.abs(esti_xx - xx))))
    print('inverse projection accuracy, yy: {}'.format(np.max(np.abs(esti_yy - yy))))
    print('inverse projection accuracy, zz: {}'.format(np.max(np.abs(esti_zz - zz))))
