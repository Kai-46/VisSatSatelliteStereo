# ===============================================================================================================
#  This file is part of Creation of Operationally Realistic 3D Environment (CORE3D).                            =
#  Copyright 2019 Cornell University - All Rights Reserved                                                      =
#  -                                                                                                            =
#  NOTICE: All information contained herein is, and remains the property of General Electric Company            =
#  and its suppliers, if any. The intellectual and technical concepts contained herein are proprietary          =
#  to General Electric Company and its suppliers and may be covered by U.S. and Foreign Patents, patents        =
#  in process, and are protected by trade secret or copyright law. Dissemination of this information or         =
#  reproduction of this material is strictly forbidden unless prior written permission is obtained              =
#  from General Electric Company.                                                                               =
#  -                                                                                                            =
#  The research is based upon work supported by the Office of the Director of National Intelligence (ODNI),     =
#  Intelligence Advanced Research Projects Activity (IARPA), via DOI/IBC Contract Number D17PC00287.            =
#  The U.S. Government is authorized to reproduce and distribute copies of this work for Governmental purposes. =
# ===============================================================================================================

import os
import numpy as np
import shutil
from colmap.read_dense import read_array
from produce_dsm import produce_dsm_from_points
from visualization.plot_height_map import plot_height_map
from visualization.save_image_only import save_image_only
import json
import logging
import multiprocessing
from coordinate_system import local_to_global
from lib.latlon_utm_converter import latlon_to_eastnorh


def convert_depth_map_worker(work_dir, out_dir, items, depth_type):
    for subdir in [out_dir, 
                   os.path.join(out_dir, 'dsm_tif'),
                   os.path.join(out_dir, 'dsm_jpg'),
                   os.path.join(out_dir, 'dsm_img_grid')]:
        if not os.path.exists(subdir):
            os.mkdir(subdir)

    mvs_dir = os.path.join(work_dir, 'colmap/mvs')
    # load inv_proj_mats
    inv_proj_mats = {}
    with open(os.path.join(mvs_dir, 'inv_proj_mats.txt')) as fp:
        for line in fp.readlines():
            tmp = line.split(' ')
            img_name = tmp[0]
            mats = np.array([float(tmp[i]) for i in range(1, 17)]).reshape((4, 4))
            inv_proj_mats[img_name] = mats

    # then read the depth maps
    depth_dir = os.path.join(mvs_dir, 'stereo/depth_maps')
    image_dir = os.path.join(mvs_dir, 'images')
    for item in items:
        idx = item.rfind('.{}.bin'.format(depth_type))
        if idx == -1:
            logging.info('something funny is happening: {}'.format(item))
            continue

        img_name = item[:idx]

        logging.info('converting depth map to dsm: {}'.format(img_name))
        
        depth_map = read_array(os.path.join(depth_dir, item))
        depth_map[depth_map <= 0.0] = np.nan    # actually it is -1e20

        # create a meshgrid
        height, width = depth_map.shape
        row, col = np.meshgrid(range(height), range(width), indexing='ij')
        col = col.reshape((1, -1))
        row = row.reshape((1, -1))
        depth = depth_map.reshape((1, -1))

        # project to scene space
        tmp = np.vstack((col, row, np.ones((1, width * height)), depth))
        tmp = np.dot(inv_proj_mats[img_name], tmp)
        tmp[0, :] /= tmp[3, :]
        tmp[1, :] /= tmp[3, :]
        tmp[2, :] /= tmp[3, :]
        height_map = tmp[2:3, :].reshape((height, width))
        valid_mask = np.logical_not(np.isnan(tmp[2, :]))
        points = tmp[:, valid_mask].T

        # convert to UTM
        lat, lon, alt = local_to_global(work_dir, points[:, 0:1], points[:, 1:2], points[:, 2:3])
        east, north = latlon_to_eastnorh(lat, lon)
        points = np.hstack((east, north, alt))

        tif_to_write = os.path.join(out_dir, 'dsm_tif', img_name[:-4] + '.tif')
        jpg_to_write = os.path.join(out_dir, 'dsm_jpg', img_name[:-4] + '.jpg')
        produce_dsm_from_points(work_dir, points, tif_to_write, jpg_to_write)

        min_val, max_val = np.nanpercentile(height_map, [1, 99])
        height_map = np.clip(height_map, min_val, max_val)
        plot_height_map(height_map, os.path.join(out_dir, 'dsm_img_grid', img_name[:-4] + '.jpg'))


def split_big_list(big_list, num_small_lists):
    cnt = len(big_list)
    indices = np.array(np.arange(cnt, dtype=np.int32))
    indices = np.array_split(indices, num_small_lists)

    small_lists = []
    for i in range(num_small_lists):
        subindices = indices[i]
        if subindices.size > 0:
            idx1 = subindices[0]
            idx2 = subindices[-1]
            small_lists.append(big_list[idx1:idx2+1])

    return small_lists


def convert_depth_maps(work_dir, out_dir, depth_type):
    mvs_dir = os.path.join(work_dir, 'colmap/mvs')
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    # all items to be converted
    depth_dir = os.path.join(mvs_dir, 'stereo/depth_maps')
    all_items = []
    for item in sorted(os.listdir(depth_dir)):
        if depth_type in item:
            all_items.append(item)

    logging.info('{} to be processed...'.format(len(all_items)))

    max_processes = 8
    all_items = split_big_list(all_items, max_processes)
    num_processes = len(all_items)    # actual number of processes to be launched

    jobs = []
    for i in range(num_processes):
        p = multiprocessing.Process(target=convert_depth_map_worker, 
                                    args=(work_dir, out_dir, all_items[i], depth_type))
        jobs.append(p)
        p.start()

    for proc in jobs:
        proc.join()
    

if __name__ == '__main__':
    work_dir = '/data2/kz298/core3d_result/aoi-d4-jacksonville'
    depth_type = 'photometric'
    convert_depth_maps(work_dir, depth_type)
