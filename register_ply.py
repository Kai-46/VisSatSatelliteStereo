import numpy as np
from lib.georegister_dense import georegister_dense
import os
import json
import glob
from lib.run_cmd import run_cmd


def register(work_dir):
    M = np.identity(3)
    t = np.zeros((1, 3))

    # add global shift
    with open(os.path.join(work_dir, 'aoi.json')) as fp:
        aoi_dict = json.load(fp)
    aoi_ll_east = aoi_dict['x']
    aoi_ll_north = aoi_dict['y'] - aoi_dict['h']
    t[0, 0] += aoi_ll_east
    t[0, 1] += aoi_ll_north

    mvs_dir = os.path.join(work_dir, 'colmap/mvs/height_maps')
    for item in sorted(glob.glob('{}/*.ply'.format(mvs_dir))):
        if 'register' not in item:
            georegister_dense(item, '{}.register.ply'.format(item),
                          os.path.join(work_dir, 'aoi.json'), M, t, filter=True)

            cmd = '/home/cornell/kz298/s2p/bin/plyflatten 0.3 {}'.format('{}.register.tif'.format(item))
            run_cmd(cmd, input='{}.register.ply'.format(item))


if __name__ == '__main__':
    work_dir = '/data2/kz298/mvs3dm_result/MasterSequesteredPark'
    register(work_dir)
