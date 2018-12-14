import colmap.read_model as read_model
import sys
import os
import numpy as np
from lib.ransac import esti_simiarity


def read_data(colmap_dir):
    points = read_model.read_points3d_binary(os.path.join(colmap_dir, 'sparse/points3D.bin'))
    points_ba = read_model.read_points3d_binary(os.path.join(colmap_dir, 'sparse_ba/points3D.bin'))

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

    # points_ba_arr is the source, with points_arr being the target
    target = np.array([points[key].xyz for key in common])
    source = np.array([points_ba[key].xyz for key in common])

    return source, target


def compute_transform(colmap_dir):
    source, target = read_data(colmap_dir)

    c, R, t = esti_simiarity(source, target)

    return c, R, t

if __name__ == '__main__':
    proj_dir = sys.argv[1]

    #proj_dir = '/data2/kz298/bak/data_aoi-d3-ucsd_pinhole/'
    #proj_dir = '/data2/kz298/bak/data_aoi-d4-jacksonville_pinhole/'

