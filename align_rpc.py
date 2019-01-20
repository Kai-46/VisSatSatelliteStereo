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
    M, t = esti_linear(source, target)

    logging.info('M:\n {}'.format(M))
    logging.info('t:\n {}'.format(t))

    return M, t


def test_align(work_dir):
    use_ransac = False

    # set log file
    log_file = os.path.join(work_dir, 'logs/log_align-sparse_norm_ba-to-rpc-linear.txt')
    # logging.basicConfig(filename=log_file, level=logging.INFO, filemode='w')
    log_hanlder = logging.FileHandler(log_file, 'w')
    log_hanlder.setLevel(logging.INFO)
    logging.root.setLevel(logging.INFO)
    logging.root.addHandler(log_hanlder)

    from datetime import datetime
    since = datetime.now()
    logging.info('Starting at {} ...'.format(since.strftime('%Y-%m-%d %H:%M:%S')))

    M, t= compute_transform(work_dir, use_ransac=use_ransac)

    ending = datetime.now()
    duration = (ending - since).total_seconds() / 60. # in minutes
    logging.info('Finishing at {}, duration: {}'.format(ending.strftime('%Y-%m-%d %H:%M:%S'), duration))

    # remove logging handler for later use
    logging.root.removeHandler(log_hanlder)


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
