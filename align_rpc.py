import os
import numpy as np
import logging
from lib.esti_similarity import esti_similarity, esti_similarity_ransac
from lib.esti_linear import esti_linear
from absolute_coordinate import triangualte_all_points


def compute_transform(work_dir, use_ransac=False):
    triangualte_all_points(work_dir)

    source = np.loadtxt(os.path.join(work_dir, 'register/normalized_coordinates.txt'))
    source = source[:, 0:3]
    target = np.loadtxt(os.path.join(work_dir, 'register/absolute_coordinates.txt'))
    target = target[:, 0:3]

    # M, t = esti_linear(source, target)

    M, t = esti_similarity(source, target)

    logging.info('M:\n {}'.format(M))
    logging.info('t:\n {}'.format(t))

    return M, t


if __name__ == '__main__':
    # import sys
    # logging.getLogger().setLevel(logging.INFO)
    # logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    #work_dir = '/data2/kz298/core3d_aoi/aoi-d4-jacksonville/'
    #work_dir = '/data2/kz298/core3d_aoi/aoi-d4-jacksonville-overlap/'

    #work_dir = '/data2/kz298/core3d_result_bak/aoi-d1-wpafb/'
    #work_dir = '/data2/kz298/core3d_result_bak/aoi-d2-wpafb/'
    #work_dir = '/data2/kz298/core3d_result_bak/aoi-d3-ucsd/'
    #work_dir = '/data2/kz298/core3d_result_bak/aoi-d4-jacksonville/'

    #work_dir = '/data2/kz298/core3d_result/aoi-d1-wpafb/'
    #work_dir = '/data2/kz298/core3d_result/aoi-d2-wpafb/'
    #work_dir = '/data2/kz298/core3d_result/aoi-d3-ucsd/'
    #work_dir = '/data2/kz298/core3d_result/aoi-d4-jacksonville/'

    work_dirs = ['/data2/kz298/core3d_result/aoi-d1-wpafb/',
                 '/data2/kz298/core3d_result/aoi-d2-wpafb/',
                 '/data2/kz298/core3d_result/aoi-d3-ucsd/',
                 '/data2/kz298/core3d_result/aoi-d4-jacksonville/']

    for work_dir in work_dirs:
        test_align(work_dir)
