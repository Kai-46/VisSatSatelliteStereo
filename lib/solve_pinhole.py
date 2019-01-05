# UPnP algorithm
import cv2
import numpy as np
from lib.check_error import check_perspective_error


def solve_pinhole(xx, yy, zz, col, row, img_size, keep_mask=None):
    diff_size = np.array([yy.size - xx.size, zz.size - xx.size, col.size - xx.size, row.size - xx.size])
    assert (np.all(diff_size == 0))

    if keep_mask is not None:
        # logging.info('discarding {} % outliers'.format((1. - np.sum(keep_mask) / keep_mask.size) * 100.))
        xx = xx[keep_mask].reshape((-1, 1))
        yy = yy[keep_mask].reshape((-1, 1))
        zz = zz[keep_mask].reshape((-1, 1))
        row = row[keep_mask].reshape((-1, 1))
        col = col[keep_mask].reshape((-1, 1))

    object_points = np.hstack((xx, yy, zz))
    image_points = np.hstack((col, row))
    cnt = image_points.shape[0]
    image_points = np.ascontiguousarray(image_points[:, :2]).reshape((cnt, 1, 2))

    width, height = img_size
    cx = width / 2.
    cy = height / 2.

    # camera_matrix = np.array([[1., 0., width / 2.],
    #                           [0., 1., height / 2.],
    #                           [0., 0., 1]])
    # dist_coeffs = np.zeros((5, 1))
    # f, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_UPNP)

    f, rvec, t = cv2.solveUPnP(object_points, image_points, cx, cy)

    R = np.zeros((3, 3))
    cv2.Rodrigues(rvec, R)

    # print('estimated focal length: {}'.format(f))
    # print('estiamted rotation matrix: {}'.format(R))
    # print('estimated translation vector: {}'.format(t))

    K = np.array([[f, 0, cx],
                  [0, f, cy],
                  [0, 0, 1]])

    return K, R, t


def rvs(dim=3):
    random_state = np.random
    H = np.eye(dim)
    D = np.ones((dim,))
    for n in range(1, dim):
        x = random_state.normal(size=(dim - n + 1,))
        D[n - 1] = np.sign(x[0])
        x[0] -= D[n - 1] * np.sqrt((x * x).sum())
        # Householder transformation
        Hx = (np.eye(dim - n + 1) - 2. * np.outer(x, x) / (x * x).sum())
        mat = np.eye(dim)
        mat[n - 1:, n - 1:] = Hx
        H = np.dot(H, mat)
        # Fix the last sign such that the determinant is 1
    D[-1] = (-1) ** (1 - (dim % 2)) * D.prod()
    # Equivalent to np.dot(np.diag(D), H) but faster, apparently
    H = (D * H.T).T
    return H


if __name__ == '__main__':
    # generate test points
    rmat = rvs()
    tvec = np.array((0, 0.5, 2)).reshape((3, 1))

    image_size = (100, 120)
    f = 100.
    intrinsic = np.array([[f, 0., image_size[0] / 2],
                         [0., f, image_size[1] / 2],
                         [0., 0., 1.]])

    print('ground-truth intrinsic: {}'.format(intrinsic))
    print('ground-truth rotation matrix: {}'.format(rmat))
    print('ground-truth translation vector: {}'.format(tvec))
    # generate 3D points
    cnt = 50
    object_points = np.random.rand(3, cnt)

    tmp = np.dot(rmat, object_points) + np.tile(tvec, (1, cnt))
    tmp = np.dot(intrinsic, tmp)

    image_points = np.vstack((tmp[0:1, :] / tmp[2:3, :], tmp[1:2] / tmp[2:3]))

    xx = object_points[0:1, :].reshape((-1, 1))
    yy = object_points[1:2, :].reshape((-1, 1))
    zz = object_points[2:3, :].reshape((-1, 1))
    col = image_points[0:1, :].reshape((-1, 1))
    row = image_points[1:2, :].reshape((-1, 1))
    K, R, t = solve_pinhole(xx, yy, zz, col, row, image_size)

    tmp = check_perspective_error(xx, yy, zz, col, row, K, R, t, keep_mask=None)
    print(tmp)
