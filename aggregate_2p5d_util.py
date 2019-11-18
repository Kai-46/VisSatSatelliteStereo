#  ===============================================================================================================
#  Copyright (c) 2019, Cornell University. All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without modification, are permitted provided that
#  the following conditions are met:
#
#      * Redistributions of source code must retain the above copyright otice, this list of conditions and
#        the following disclaimer.
#
#      * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and
#        the following disclaimer in the documentation and/or other materials provided with the distribution.
#
#      * Neither the name of Cornell University nor the names of its contributors may be used to endorse or
#        promote products derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
#  WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
#  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE
#  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
#  TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
#  HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
#   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
#  OF SUCH DAMAGE.
#
#  Author: Kai Zhang (kz298@cornell.edu)
#
#  The research is based upon work supported by the Office of the Director of National Intelligence (ODNI),
#  Intelligence Advanced Research Projects Activity (IARPA), via DOI/IBC Contract Number D17PC00287.
#  The U.S. Government is authorized to reproduce and distribute copies of this work for Governmental purposes.
#  ===============================================================================================================


import os
import numpy as np
import shutil
from colmap.read_dense import read_array
from produce_dsm import produce_dsm_from_points
from visualization.plot_height_map import plot_height_map
import logging
import multiprocessing
from coordinate_system import local_to_global
from lib.latlon_utm_converter import latlon_to_eastnorh


def convert_depth_map_worker(work_dir, out_dir, item, depth_type):
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

    idx = item.rfind('.{}.bin'.format(depth_type))
    if idx == -1:
        logging.info('something funny is happening: {}'.format(item))
        return

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


def convert_depth_maps(work_dir, out_dir, depth_type, max_processes=-1):
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
    if max_processes <= 0:
        max_processes = multiprocessing.cpu_count()

    pool = multiprocessing.Pool(min(max_processes, len(all_items)))
    for item in all_items:
        pool.apply_async(convert_depth_map_worker, args=(work_dir, out_dir, item, depth_type))

    pool.close()
    pool.join()


if __name__ == '__main__':
    pass
