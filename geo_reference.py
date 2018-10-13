import lib.read_model as read_model
import sys
import os
import numpy as np
from lib.plyfile import PlyData
import json


def esti_linear_map(proj_dir):
    points = read_model.read_points3d_binary(os.path.join(proj_dir, 'sparse/points3D.bin'))
    points_ba = read_model.read_points3d_binary(os.path.join(proj_dir, 'sparse_ba/points3D.bin'))

    points_key = set(points.keys())
    points_ba_key = set(points_ba.keys())

    common = points_key & points_ba_key
    common = list(common)

    cnt = len(common)
    sample_cnt = np.inf
    if cnt > sample_cnt:
        indices = np.random.permutation(cnt)
        common = [common[x] for x in indices[:sample_cnt]]
    else:
        sample_cnt = cnt

    points_arr = np.array([points[key].xyz for key in common])
    points_ba_arr = np.array([points_ba[key].xyz for key in common])

    all_zeros = np.zeros((sample_cnt, 1))
    all_ones = np.ones((sample_cnt, 1))

    A1 = np.hstack((points_ba_arr, np.tile(all_zeros, (1, 6)), all_ones, np.tile(all_zeros, (1, 2))))
    A2 = np.hstack((np.tile(all_zeros, (1, 3)), points_ba_arr, np.tile(all_zeros, (1, 3)), all_zeros, all_ones, all_zeros))
    A3 = np.hstack((np.tile(all_zeros, (1, 6)), points_ba_arr, np.tile(all_zeros, (1, 2)), all_ones))

    A = np.vstack((A1, A2, A3))
    b = np.vstack((points_arr[:, 0:1], points_arr[:, 1:2], points_arr[:, 2:3]))

    result = np.linalg.lstsq(A, b, rcond=None)[0]
    M = result[0:9].reshape((3, 3))
    t = result[9:12].reshape((3, 1))

    # check whether M is close to a rotation matrix
    u, s, vh = np.linalg.svd(M)
    # print(M)
    # print('singular values of M: {}'.format(s))
    cond = s[0] / s[2]
    print('condition number of M: {}, smallest singular value: {}'.format(cond, s[2]))
    assert(s[2] > 0 and cond < 1.5)

    # check the MSE after applying the linear map
    points_ba_reg = np.dot(points_ba_arr, M.T) + np.tile(t.T, (sample_cnt, 1))
    mse_before = np.mean((points_arr - points_ba_arr)**2)
    mse_after = np.mean((points_arr - points_ba_reg)**2)
    print('mse before: {}'.format(mse_before))
    print('mse after: {}'.format(mse_after))

    return M, t


if __name__ == '__main__':
    proj_dir = sys.argv[1]
    # proj_dir = 'data_jacksonville_pinhole'

    M, t = esti_linear_map(proj_dir)

    # map the dense point cloud into absolute coordinate frame
    dense = PlyData.read(os.path.join(proj_dir, 'dense/fused.ply'))

    points = np.hstack((dense['vertex']['x'].reshape((-1, 1)),
                        dense['vertex']['y'].reshape((-1, 1)),
                        dense['vertex']['z'].reshape((-1, 1))))
    normals = np.hstack((dense['vertex']['nx'].reshape((-1, 1)),
                         dense['vertex']['ny'].reshape((-1, 1)),
                         dense['vertex']['nz'].reshape((-1, 1))))

    points_reg = np.dot(points, M.T) + np.tile(t.T, (points.shape[0], 1))
    normals_reg = np.dot(normals, M.T) + np.tile(t.T, (normals.shape[0], 1))

    # read json file
    with open(os.path.join(proj_dir, 'roi.json')) as fp:
        roi = json.load(fp)
    comment_1 = 'zone_number: {}, zone_letter: {}'.format(roi['zone_number'], roi['zone_letter'])
    comment_2 = 'x, y, w, h : {}, {}, {}, {}'.format(roi['x'], roi['y'], roi['w'], roi['h'])
    print(comment_1)
    print(comment_2)

    dense.comments = [comment_1, comment_2]

    # write to plydata object
    # in perspective camera approximation, the world coordinate frame is (south, east, above)
    # the UTM coordinate frame is (east, north, above)
    dense['vertex']['x'] = points_reg[:, 1] + roi['x']
    dense['vertex']['y'] = roi['y'] - points_reg[:, 0]
    dense['vertex']['z'] = points_reg[:, 2]

    dense['vertex']['nx'] = normals_reg[:, 1]
    dense['vertex']['ny'] = normals_reg[:, 0]
    dense['vertex']['nz'] = normals_reg[:, 2]

    dense.write(os.path.join(proj_dir, 'dense/fused_geo_referenced.ply'))

    # print('hello world!')