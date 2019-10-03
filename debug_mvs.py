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
import json
from colmap.read_dense import read_array
from produce_dsm import produce_dsm_from_points
from visualization.plot_height_map import plot_height_map
from visualization.save_image_only import save_image_only
from coordinate_system import local_to_global
from lib.latlon_utm_converter import latlon_to_eastnorh


def convert_depth_map(mvs_dir, out_dir, item, depth_type):
    for subdir in [out_dir, 
                   os.path.join(out_dir, 'dsm_tif'),
                   os.path.join(out_dir, 'dsm_jpg'),
                   os.path.join(out_dir, 'dsm_img_grid')]:
        if not os.path.exists(subdir):
            os.mkdir(subdir)

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

    idx = item.rfind('.{}.bin'.format(depth_type))
    if idx == -1:
        logging.info('something funny is happening: {}'.format(item))
        return
    img_name = item[:idx]

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
    lat, lon, alt = local_to_global(mvs_dir, points[:, 0:1], points[:, 1:2], points[:, 2:3])
    east, north = latlon_to_eastnorh(lat, lon)
    points = np.hstack((east, north, alt))

    tif_to_write = os.path.join(out_dir, 'dsm_tif', img_name[:-4] + '.tif')
    jpg_to_write = os.path.join(out_dir, 'dsm_jpg', img_name[:-4] + '.jpg')
    produce_dsm_from_points(mvs_dir, points, tif_to_write, jpg_to_write)

    min_val, max_val = np.nanpercentile(height_map, [1, 99])
    height_map = np.clip(height_map, min_val, max_val)
    plot_height_map(height_map, os.path.join(out_dir, 'dsm_img_grid', img_name[:-4] + '.jpg'))


def debug_mvs(colmap_work_dir, out_dir, ref_img_name, src_img_names):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # prepare directory structure for stereo
    fnames_to_link = ['images', 'sparse', 'last_rows.txt', 'proj_mats.txt', 'inv_proj_mats.txt', 'depth_ranges.txt']
    for fname in fnames_to_link:
        if os.path.exists(os.path.join(out_dir, fname)):
            os.unlink(os.path.join(out_dir, fname))
        os.symlink(os.path.relpath(os.path.join(colmap_work_dir, 'colmap/mvs', fname), out_dir),
                  os.path.join(out_dir, fname))

    if os.path.exists(os.path.join(out_dir, 'aoi.json')):
        os.unlink(os.path.join(out_dir, 'aoi.json'))
    os.symlink(os.path.relpath(os.path.join(colmap_work_dir, 'aoi.json'), out_dir),
               os.path.join(out_dir, 'aoi.json'))

    stereo_subdir = os.path.join(out_dir, 'stereo')
    for subdir in [stereo_subdir,
                   os.path.join(stereo_subdir, 'consistency_graphs'),
                   os.path.join(stereo_subdir, 'depth_maps'),
                   os.path.join(stereo_subdir, 'normal_maps')]:
        if not os.path.exists(subdir):
            os.mkdir(subdir)

    with open(os.path.join(stereo_subdir, 'patch-match.cfg'), 'w') as fp:
        fp.write(ref_img_name + '\n')
        fp.write(','.join(src_img_names) + '\n')

    # run colmap MVS
    window_radius = 3
    gpu_index = -1
    cmd = 'colmap patch_match_stereo --workspace_path {} \
                    --PatchMatchStereo.window_radius {}\
                    --PatchMatchStereo.min_triangulation_angle 5.0 \
                    --PatchMatchStereo.filter 1 \
                    --PatchMatchStereo.filter_min_triangulation_angle 4.99 \
                    --PatchMatchStereo.filter_min_ncc 0.1 \
                    --PatchMatchStereo.geom_consistency 0 \
                    --PatchMatchStereo.gpu_index={} \
                    --PatchMatchStereo.num_samples 15 \
                    --PatchMatchStereo.num_iterations 12 \
                    --PatchMatchStereo.overwrite 1'.format(out_dir,
                                                           window_radius, gpu_index)
    os.system(cmd)

    # convert depth map
    depth_type = 'photometric'
    for item in sorted(os.listdir(os.path.join(out_dir, 'stereo/depth_maps'))):
        if depth_type not in item:
            continue
        convert_depth_map(out_dir, os.path.join(out_dir, 'dsm'), item, depth_type)


if __name__ == '__main__':
    colmap_work_dir = '/data2/kz298/mvs3dm_result_paper/Explorer'
    out_dir = '/data2/kz298/mvs3dm_result_paper/debug_mvs_test'
    ref_img_name = '0000_WV03_14NOV15_135121-P1BS-500171606160_05_P005.png'
    src_img_names = ['0001_WV03_15JAN05_135727-P1BS-500497282040_01_P001.png', ]
    debug_mvs(colmap_work_dir, out_dir, ref_img_name, src_img_names)