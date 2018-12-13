import lib.read_model as read_model
import sys
import os
import numpy as np
from lib.plyfile import PlyData, PlyElement
import json
import lib.procrustes as procrustes


def esti_transform(proj_dir):
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

    d, z, tform = procrustes.procrustes(points_arr, points_ba_arr, reflection=False)   

    M = tform['scale'] * tform['rotation'].T
    t = tform['translation'].reshape((-1, 1))
    print('shape of M: {}'.format(M.shape))
    print('shape of t: {}'.format(t.shape))

    # check whether M is close to a rotation matrix
    u, s, vh = np.linalg.svd(M)
    # print(M)
    # print('singular values of M: {}'.format(s))
    cond = s[0] / s[2]
    print('condition number of M: {}, smallest singular value: {}'.format(cond, s[2]))
    assert(s[2] > 0 and cond < 1.1)

    # check the MSE after applying the linear map
    points_ba_reg = np.dot(points_ba_arr, M.T) + np.tile(t.T, (sample_cnt, 1))
    mse_before = np.mean(np.sum((points_arr - points_ba_arr)**2, axis=1))
    mse_after = np.mean(np.sum((points_arr - points_ba_reg)**2, axis=1))
    print('rmse before: {}'.format(np.sqrt(mse_before)))
    print('rmse after: {}'.format(np.sqrt(mse_after)))

    # plot absolute deviation
    import matplotlib.pyplot as plt
    plt.subplot(121)
    std_dev = np.sqrt(np.sum((points_arr - points_ba_arr)**2, axis=1))
    plt.hist(std_dev, bins=100, density=True)
    plt.title('before procrustes')
    plt.xlabel('Euclidean distance')
    plt.ylabel('density')

    plt.subplot(122)
    std_dev = np.sqrt(np.sum((points_arr -points_ba_reg)**2, axis=1))
    plt.hist(std_dev, bins=100, density=True)
    plt.title('after procrustes')
    plt.xlabel('Euclidean distance')
    plt.ylabel('density')

    plt.show()

    return M, t

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
    
    # M = s[1] * np.dot(u, vh)
    # M = np.dot(np.dot(u, np.diag(s)), vh)

    # check the MSE after applying the linear map
    points_ba_reg = np.dot(points_ba_arr, M.T) + np.tile(t.T, (sample_cnt, 1))
    mse_before = np.mean(np.sum((points_arr - points_ba_arr)**2, axis=1))
    mse_after = np.mean(np.sum((points_arr - points_ba_reg)**2, axis=1))
    print('rmse before: {}'.format(np.sqrt(mse_before)))
    print('rmse after: {}'.format(np.sqrt(mse_after)))

    return M, t


if __name__ == '__main__':
    proj_dir = sys.argv[1]

    # proj_dir = 'data_aoi-d1-wpafb_pinhole'
    #proj_dir = 'data_aoi-d2-wpafb_pinhole'
    # proj_dir = 'data_aoi-d3-ucsd_pinhole'
    #proj_dir = 'data_aoi-d4-jacksonville_pinhole'

    # M, t = esti_linear_map(proj_dir)
    M, t = esti_transform(proj_dir)

    # map the dense point cloud into absolute coordinate frame
    dense = PlyData.read(os.path.join(proj_dir, 'dense/fused.ply'))

    points = np.hstack((dense['vertex']['x'].reshape((-1, 1)),
                        dense['vertex']['y'].reshape((-1, 1)),
                        dense['vertex']['z'].reshape((-1, 1))))
    normals = np.hstack((dense['vertex']['nx'].reshape((-1, 1)),
                         dense['vertex']['ny'].reshape((-1, 1)),
                         dense['vertex']['nz'].reshape((-1, 1))))
    colors = np.hstack((dense['vertex']['red'].reshape((-1, 1)),
                        dense['vertex']['green'].reshape((-1, 1)),
                        dense['vertex']['blue'].reshape((-1, 1))))

    points_reg = np.dot(points, M.T) + np.tile(t.T, (points.shape[0], 1))
    normals_reg = np.dot(normals, M.T) + np.tile(t.T, (normals.shape[0], 1))

    z = points_reg[:, 2]
    below_thres = np.percentile(z, 0)
    above_thres = np.percentile(z, 100)
    print('below_thres: {}, above_thres: {}'.format(below_thres, above_thres))

    mask = np.logical_and(z>=below_thres, z<=above_thres)
    #mask = np.tile(mask.reshape((-1, 1)), (1, 3))

    points_reg = points_reg[mask, :]
    normals_reg = normals_reg[mask, :]
    colors = colors[mask, :]

    # read json file
    with open(os.path.join(proj_dir, 'roi.json')) as fp:
        roi = json.load(fp)
    # comment_1 = 'zone_number: {}, zone_letter: {}'.format(roi['zone_number'], roi['zone_letter'])
    comment_1 = 'projection: UTM {}{}'.format(roi['zone_number'], roi['zone_letter'])
    comment_2 = 'x, y, w, h : {}, {}, {}, {}'.format(roi['x'], roi['y'], roi['w'], roi['h'])
    print(comment_1)
    print(comment_2)

    comments = [comment_1, comment_2]

    # write to plydata object
    # in perspective camera approximation, the world coordinate frame is (south, east, above)
    # the UTM coordinate frame is (east, north, above)
    x = points_reg[:, 1:2] + roi['x']
    y = roi['y'] - points_reg[:, 0:1]
    z = points_reg[:, 2:3]

    nx = normals_reg[:, 1:2]
    ny = normals_reg[:, 0:1]
    nz = normals_reg[:, 2:3]

    points = np.hstack((x, y, z, nx, ny, nz, colors))
    vertex = np.array([tuple(point) for point in points], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), 
                                                                 ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'), 
                                                                 ('red', 'uint8'), ('green', 'uint8'), ('blue', 'uint8')])
    el = PlyElement.describe(vertex, 'vertex')
    PlyData([el], byte_order='<', comments=comments).write(os.path.join(proj_dir, 'dense/fused_geo_referenced.ply'))

    # create a ply in s2p format
    # points = np.hstack((dense['vertex']['x'].reshape((-1, 1)),
    #                     dense['vertex']['y'].reshape((-1, 1)),
    #                     dense['vertex']['z'].reshape((-1, 1)),
    #                     dense['vertex']['red'].reshape((-1, 1)),
    #                     dense['vertex']['green'].reshape((-1, 1)),
    #                     dense['vertex']['blue'].reshape((-1, 1))))
    # vertex = np.array([tuple(point) for point in points], dtype=[('x', 'f8'), ('y', 'f8'), ('z', 'f8'), 
    #                                                              ('red', 'uint8'), ('green', 'uint8'), ('blue', 'uint8')])
    # el = PlyElement.describe(vertex, 'vertex')
    # PlyData([el], text=True, comments=['created by Kai', comment_1]).write(os.path.join(proj_dir, 'dense/s2p_format.ply'))
    #PlyData([el], byte_order='<', comments=['created by Kai', comment_1]).write(os.path.join(proj_dir, 'dense/s2p_format.ply'))
    
    # print('hello world!')
