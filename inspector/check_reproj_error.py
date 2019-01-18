import numpy as np
import quaternion
from colmap.extract_sfm import extract_sfm
from inspector.plot_reproj_err import plot_reproj_err
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def check_reproj_error(sparse_dir, model):
    assert(model == 'pinhole' or model == 'perspective')

    camera_params, all_tracks = extract_sfm(sparse_dir)

    # construct projection matrix for each camera
    proj_matrices = {}
    img_sizes = {}
    for img_name in camera_params.keys():
        size, intrinsic, qvec, tvec = camera_params[img_name]

        img_sizes[img_name] = size

        if model == 'pinhole':
            K = np.array([[intrinsic[0], 0., intrinsic[2]],
                          [ 0., intrinsic[1], intrinsic[3]],
                          [ 0., 0., 1.]])
        else:
            K = np.array([[intrinsic[0], intrinsic[4], intrinsic[2]],
                          [0., intrinsic[1], intrinsic[3]],
                          [0., 0., 1.]])
        R = quaternion.as_rotation_matrix(np.quaternion(qvec[0], qvec[1], qvec[2], qvec[3]))
        t = np.reshape(tvec, (3, 1))

        proj_matrices[img_name] = np.dot(K, np.hstack((R, t)))

    my_errs = []
    colmap_errs = []
    for track in all_tracks:
        xyz = np.array([track['xyz'][0], track['xyz'][1], track['xyz'][2], 1.]).reshape((4, 1))
        err = 0.
        cnt = 0
        for pixel in track['pixels']:
            img_name, col, row = pixel

            # check whether there are out-of-boundary key points
            width, height = img_sizes[img_name]
            if col < 0 or col > width or row < 0 or row > height:
                continue

            tmp = np.dot(proj_matrices[img_name], xyz)
            esti_col = tmp[0] / tmp[2]
            esti_row = tmp[1] / tmp[2]

            sq_err = (col - esti_col) ** 2 + (row - esti_row) ** 2
            err += np.sqrt(sq_err)
            cnt += 1
        if cnt > 1:
            err /= cnt
        # check whether it agrees with what colmap computes
        # if np.abs(err - track['err']) > 1e-1:
        #     print(track)
        #     print('err: {}'.format(err))
        #     exit(-1)

            my_errs.append(err)
            colmap_errs.append(track['err'])

    my_errs = np.array(my_errs)
    colmap_errs = np.array(colmap_errs)

    print('my_errs, median: {}, mean: {}'.format(np.median(my_errs), np.mean(my_errs)))
    print('colmap_errs, median: {}, mean: {}'.format(np.median(colmap_errs), np.mean(colmap_errs)))

    # plot_reproj_err(my_errs, os.path.join(sparse_dir, 'sfm_reproj_err_my.jpg'))
    # plot_reproj_err(colmap_errs, os.path.join(sparse_dir, 'sfm_reproj_err_colmap.jpg'))


if __name__ == '__main__':
    #sparse_dir = '/data2/kz298/mvs3dm_result/Explorer/colmap/sparse_for_mvs'
    sparse_dir = '/data2/kz298/mvs3dm_result/Explorer/colmap/dense/sparse'
    model = 'pinhole'
    check_reproj_error(sparse_dir, model)