import colmap.read_model as read_model
import sys
import os
import numpy as np

from lib.esti_similarity import esti_similarity, esti_similarity_ransac


def read_data(colmap_dir):
    points = read_model.read_points3d_binary(os.path.join(colmap_dir, 'sparse/points3D.bin'))
    points_ba = read_model.read_points3d_binary(os.path.join(colmap_dir, 'sparse_ba/points3D.bin'))

    points_key = set(points.keys())
    points_ba_key = set(points_ba.keys())

    common = points_key & points_ba_key
    common = list(common)

    # cnt = len(common)

    # take a subset of all the 3D points
    # sample_cnt = np.inf
    # if cnt > sample_cnt:
    #     indices = np.random.permutation(cnt)
    #     common = [common[x] for x in indices[:sample_cnt]]
    # else:
    #     sample_cnt = cnt

    # select small reproj. err ones
    common = [key for key in common if points[key].error < 1. and points_ba[key].error < 1.]

    # points_ba_arr is the source, with points_arr being the target
    target = np.array([points[key].xyz for key in common])
    source = np.array([points_ba[key].xyz for key in common])

    return source, target


def compute_transform(work_dir, use_ransac=False):
    colmap_dir = os.path.join(work_dir, 'colmap')
    source, target = read_data(colmap_dir)

    if use_ransac:
        c, R, t = esti_similarity_ransac(source, target)
    else:
        c, R, t = esti_similarity(source, target)

    return c, R, t

if __name__ == '__main__':
    import logging

    # logging.getLogger().setLevel(logging.INFO)
    # logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    #work_dir = '/data2/kz298/core3d_aoi/aoi-d4-jacksonville/'
    # work_dir = '/data2/kz298/core3d_aoi/aoi-d4-jacksonville-overlap/'

    #work_dir = '/data2/kz298/core3d_result_bak/aoi-d1-wpafb/'
    #work_dir = '/data2/kz298/core3d_result_bak/aoi-d2-wpafb/'
    #work_dir = '/data2/kz298/core3d_result_bak/aoi-d3-ucsd/'
    #work_dir = '/data2/kz298/core3d_result_bak/aoi-d4-jacksonville/'

    # work_dir = '/data2/kz298/core3d_result/aoi-d1-wpafb/'
    #work_dir = '/data2/kz298/core3d_result/aoi-d2-wpafb/'
    #work_dir = '/data2/kz298/core3d_result/aoi-d3-ucsd/'
    work_dir = '/data2/kz298/core3d_result/aoi-d4-jacksonville/'

    use_ransac = False
    log_file = os.path.join(work_dir, 'logs/log_align_sparse_ransac_{}.txt'.format(use_ransac))
    logging.basicConfig(filename=log_file, level=logging.INFO, filemode='w')

    from datetime import datetime
    logging.info('Starting at {} ...'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

    c, R, t = compute_transform(work_dir, use_ransac=use_ransac)

    logging.info('Finishing at {} ...'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))