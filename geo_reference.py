import lib.read_model as read_model
import sys
import os
import numpy as np
from lib.plyfile import PlyData, PlyElement
import json
from lib.ransac import esti_simiarity
import matplotlib.pyplot as plt


def read_data(proj_dir):
    points = read_model.read_points3d_binary(os.path.join(proj_dir, 'sparse/points3D.bin'))
    points_ba = read_model.read_points3d_binary(os.path.join(proj_dir, 'sparse_ba/points3D.bin'))

    points_key = set(points.keys())
    points_ba_key = set(points_ba.keys())

    common = points_key & points_ba_key
    common = list(common)

    cnt = len(common)

    # take a subset of all the 3D points
    # sample_cnt = np.inf
    # if cnt > sample_cnt:
    #     indices = np.random.permutation(cnt)
    #     common = [common[x] for x in indices[:sample_cnt]]
    # else:
    #     sample_cnt = cnt

    points_arr = np.array([points[key].xyz for key in common])
    points_ba_arr = np.array([points_ba[key].xyz for key in common])

    # points_ba_arr is the source, with points_arr being the target
    return points_ba_arr, points_arr


def inspect_align(source, target, source_aligned, out_dir):
    mse_before = np.mean(np.sum((source - target)**2, axis=1))
    mse_after = np.mean(np.sum((source_aligned - target)**2, axis=1))
    print('rmse before: {}'.format(np.sqrt(mse_before)))
    print('rmse after: {}'.format(np.sqrt(mse_after)))

    # plot absolute deviation
    plt.figure(figsize=(8, 4))
    plt.subplot(121)
    std_dev = np.sqrt(np.sum((source - target)**2, axis=1))
    plt.hist(std_dev, bins=100, density=True)
    plt.title('before procrustes')
    plt.xlabel('Euclidean distance')
    plt.ylabel('density')

    plt.subplot(122)
    std_dev = np.sqrt(np.sum((source_aligned - target)**2, axis=1))

    max_std_dev = np.max(std_dev)
    thres = 10 if max_std_dev > 10 else max_std_dev
    mask = std_dev <= thres
    ratio = (1. - np.sum(mask) / mask.size) * 100.
    std_dev = std_dev[mask]

    plt.hist(std_dev, bins=100, density=True)
    plt.title('after procrustes\n{:.2f}% points above thres {}'.format(ratio, thres))
    plt.xlabel('Euclidean distance (meter)')
    plt.ylabel('density')
    plt.xticks(range(0, 11))
    plt.tight_layout()

    plt.savefig(out_dir)


def align(proj_dir):
    points_ba_arr, points_arr = read_data(proj_dir)

    c, R, t = esti_simiarity(points_ba_arr, points_arr)

    # inspect the alignment
    points_ba_reg = np.dot(points_ba_arr, c * R) + np.tile(t, (points_ba_arr.shape[0], 1))
    out_dir = os.path.join(proj_dir, 'dense/align_residual.png')
    inspect_align(points_ba_arr, points_arr, points_ba_reg, out_dir)

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

    points_reg = np.dot(points, c * R) + np.tile(t, (points.shape[0], 1))
    normals_reg = np.dot(normals, c * R) + np.tile(t, (normals.shape[0], 1))

    # z = points_reg[:, 2]
    # below_thres = np.percentile(z, 0)
    # above_thres = np.percentile(z, 100)
    # print('below_thres: {}, above_thres: {}'.format(below_thres, above_thres))
    #
    # mask = np.logical_and(z>=below_thres, z<=above_thres)
    # #mask = np.tile(mask.reshape((-1, 1)), (1, 3))
    #
    # points_reg = points_reg[mask, :]
    # normals_reg = normals_reg[mask, :]
    # colors = colors[mask, :]

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


if __name__ == '__main__':
    proj_dir = sys.argv[1]

    #proj_dir = '/data2/kz298/bak/data_aoi-d3-ucsd_pinhole/'
    #proj_dir = '/data2/kz298/bak/data_aoi-d4-jacksonville_pinhole/'
    align(proj_dir)
