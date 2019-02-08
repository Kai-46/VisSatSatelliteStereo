import os
import numpy as np
import logging
from lib.esti_similarity import esti_similarity, esti_similarity_ransac
from lib.esti_linear import esti_linear
import json


def check_align(work_dir, source, target):
    with open(os.path.join(work_dir, 'aoi.json')) as fp:
        aoi_dict = json.load(fp)
    aoi_ll_east = aoi_dict['x']
    aoi_ll_north = aoi_dict['y'] - aoi_dict['h']

    east = source[:, 0:1] + aoi_ll_east
    north = source[:, 1:2] + aoi_ll_north
    height = source[:, 2:3]
    source = np.hstack((east, north, height))

    logging.info('\nsource --> target: none')
    err = np.sqrt(np.sum((source - target) ** 2, axis=1))
    logging.info('min: {}, max: {}, mean: {}, median: {}'.format(np.min(err), np.max(err), np.mean(err), np.median(err)))

    logging.info('\nsource --> target: linear')
    M, t = esti_linear(source, target)

    logging.info('\nsource --> target: similarity')
    M, t = esti_similarity(source, target)

    logging.info('\nsource --> target: translation')
    t = np.mean(target - source, axis=0).reshape((1, 3))
    logging.info('translation: {}'.format(t))
    source_aligned = source + np.tile(t, (source.shape[0], 1))
    err = np.sqrt(np.sum((source_aligned - target) ** 2, axis=1))
    logging.info('min: {}, max: {}, mean: {}, median: {}'.format(np.min(err), np.max(err), np.mean(err), np.median(err)))


if __name__ == '__main__':
    work_dirs = ['/data2/kz298/mvs3dm_result/Explorer',
                '/data2/kz298/mvs3dm_result/MasterProvisional1',
                '/data2/kz298/mvs3dm_result/MasterProvisional2',
                '/data2/kz298/mvs3dm_result/MasterProvisional3',
                '/data2/kz298/mvs3dm_result/MasterSequestered1',
                '/data2/kz298/mvs3dm_result/MasterSequestered2',
                '/data2/kz298/mvs3dm_result/MasterSequestered3',
                '/data2/kz298/mvs3dm_result/MasterSequesteredPark']
    #
    work_dirs = [work_dirs[0], ]

    from lib.logger import GlobalLogger
    logger = GlobalLogger()
    logger.turn_on_terminal()

    for work_dir in work_dirs:
        colmap_dir = os.path.join(work_dir, 'colmap')
        for dir in [os.path.join(colmap_dir, 'sfm_perspective/init_inspect'),
                    os.path.join(colmap_dir, 'sfm_perspective/corrected_inspect'),
                    os.path.join(colmap_dir, 'sfm_perspective/corrected_triangulate_ba_inspect')
                    ]:
            print('\n\n{}'.format(dir))
            source = np.loadtxt(os.path.join(dir, 'sfm_coordinates.txt'))
            target = np.loadtxt(os.path.join(dir, 'rpc_absolute_coordinates.txt'))
            source = source[:, 0:3]
            target = target[:, 0:3]

            # logger.set_log_file(os.path.join(dir, 'log_align_similarity.txt'))
            # M, t = esti_similarity(source, target)
            # logger.set_log_file(os.path.join(dir, 'log_align_linear.txt'))
            # M, t = esti_linear(source, target)

            # logger.set_log_file(os.path.join(dir, 'log_perspective_triangulation.txt'))
            # compute absolute error
            with open(os.path.join(work_dir, 'aoi.json')) as fp:
                aoi_dict = json.load(fp)
            aoi_ll_east = aoi_dict['x']
            aoi_ll_north = aoi_dict['y'] - aoi_dict['h']

            east = source[:, 0:1] + aoi_ll_east
            north = source[:, 1:2] + aoi_ll_north
            height = source[:, 2:3]
            source = np.hstack((east, north, height))

            # if we change target to a local
            # south = aoi_dict['y'] - target[:, 1:2]
            # east = target[:, 0:1] - aoi_dict['x']
            # height = target[:, 2:3]
            # target = np.hstack((south, east, height))

            err = np.sqrt(np.sum((source - target)**2, axis=1))
            logging.info('\ntriangulation err, min, mean, median, max: {}, {}, {}, {}'.format(np.min(err), np.mean(err), np.median(err), np.max(err)))

            # err = np.sqrt(np.sum((source_utm - target)**2, axis=1))
            # logging.info('\ntriangulation err, min, mean, median, max: {}, {}, {}, {}'.format(np.min(err), np.mean(err), np.median(err), np.max(err)))

            # logging.info('\nsource --> source_utm: linear')
            # M, t = esti_linear(source, source_utm)
            #
            # logging.info('\nsource --> source_utm: similarity')
            # M, t = esti_similarity(source, source_utm)

            # logging.info('\nmanually compute source_utm --> source: linear')
            # M_inv = np.linalg.inv(M)
            # t_hat = np.dot(-t, M_inv)
            # logging.info('M: {}'.format(M_inv))
            # logging.info('t: {}'.format(t_hat))
            # source_utm_aligned = np.dot(source_utm, M_inv) + np.tile(t_hat, (source_utm.shape[0], 1))
            # err = np.sqrt(np.sum((source_utm_aligned - source)**2, axis=1))
            # logging.info('manual err, min, mean, median, max: {}, {}, {}, {}'.format(np.min(err), np.mean(err), np.median(err), np.max(err)))

            # logging.info('\nsource_utm --> source: linear')
            # M, t = esti_linear(source_utm, source)
            #
            # logging.info('\nsource_utm --> source: similarity')
            # M, t = esti_similarity(source_utm, source)

            logging.info('\nsource --> target: linear')
            M, t = esti_linear(source, target)

            logging.info('\nsource --> target: similarity')
            M, t = esti_similarity(source, target)

            logging.info('\nsource --> target: translation')
            t = np.mean(target - source, axis=0).reshape((1, 3))
            logging.info('translation: {}'.format(t))
            source_aligned = source + np.tile(t, (source.shape[0], 1))
            err = np.sqrt(np.sum((source_aligned - target)**2, axis=1))
            logging.info('err, min, mean, median, max: {}, {}, {}, {}'.format(np.min(err), np.mean(err), np.median(err), np.max(err)))


            # logging.info('\nsource_utm --> target: linear')
            # M, t = esti_linear(source_utm, target)
            #
            # logging.info('\nsource_utm --> target: similarity')
            # M, t = esti_similarity(source_utm, target)

