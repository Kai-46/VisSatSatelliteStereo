import os
import numpy as np
import logging
from lib.esti_similarity import esti_similarity, esti_similarity_ransac
from lib.esti_linear import esti_linear
from absolute_coordinate import triangualte_all_points


def compute_transform(work_dir, use_ransac=False):
#    triangualte_all_points(work_dir)

    source = np.loadtxt(os.path.join(work_dir, 'register/normalized_coordinates.txt'))
    target = np.loadtxt(os.path.join(work_dir, 'register/absolute_coordinates.txt'))

    source = source[:, 0:3]
    target = target[:, 0:3]

    # M, t = esti_linear(source, target)
    M, t = esti_similarity(source, target)

    return M, t


if __name__ == '__main__':
    from lib.logger import GlobalLogger
    logger = GlobalLogger()
    logger.turn_on_terminal()

    work_dirs = ['/data2/kz298/mvs3dm_result/Explorer',
                '/data2/kz298/mvs3dm_result/MasterProvisional1',
                '/data2/kz298/mvs3dm_result/MasterProvisional2',
                '/data2/kz298/mvs3dm_result/MasterProvisional3',
                '/data2/kz298/mvs3dm_result/MasterSequestered1',
                '/data2/kz298/mvs3dm_result/MasterSequestered2',
                '/data2/kz298/mvs3dm_result/MasterSequestered3',
                '/data2/kz298/mvs3dm_result/MasterSequesteredPark']

    compute_transform(work_dirs[0])