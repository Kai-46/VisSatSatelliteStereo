import numpy as np
from pyquaternion import Quaternion


def check_reproj_error(camera_params, all_tracks, camera_model):
    assert(camera_model == 'PINHOLE' or camera_model == 'PERSPECTIVE')

    # construct projection matrix for each camera
    proj_matrices = {}
    for img_name in camera_params.keys():
        img_size, intrinsic, qvec, tvec = camera_params[img_name]
        if camera_model == 'PINHOLE':
            K = np.array([[intrinsic[0], 0., intrinsic[2]],
                          [ 0., intrinsic[1], intrinsic[3]],
                          [ 0., 0., 1.]])
        else:
            K = np.array([[intrinsic[0], intrinsic[4], intrinsic[2]],
                          [0., intrinsic[1], intrinsic[3]],
                          [0., 0., 1.]])
        R = Quaternion(qvec[0], qvec[1], qvec[2], qvec[3]).rotation_matrix
        t = np.reshape(tvec, (3, 1))

        proj_matrices[img_name] = np.dot(K, np.hstack((R, t)))

    my_errs = []
    colmap_errs = []
    for track in all_tracks:
        xyz = np.array([track['xyz'][0], track['xyz'][1], track['xyz'][2], 1.]).reshape((4, 1))
        err = 0.
        for pixel in track['pixels']:
            img_name, col, row = pixel
            tmp = np.dot(proj_matrices[img_name], xyz)
            esti_col = tmp[0] / tmp[2]
            esti_row = tmp[1] / tmp[2]

            sq_err = (col - esti_col) ** 2 + (row - esti_row) ** 2
            err += np.sqrt(sq_err)
        err /= len(track['pixels'])

        my_errs.append(err)
        colmap_errs.append(track['err'])

    return np.array(my_errs), np.array(colmap_errs)


if __name__ == '__main__':
    #sparse_dir = '/data2/kz298/mvs3dm_result/Explorer/colmap/sparse_for_mvs'
    sparse_dir = '/data2/kz298/mvs3dm_result/Explorer/colmap/dense/sparse'
    model = 'pinhole'
    check_reproj_error(sparse_dir, model)