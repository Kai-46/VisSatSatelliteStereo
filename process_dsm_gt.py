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
import json
import numpy as np
from lib.dsm_util import read_dsm_tif
from lib.latlon_utm_converter import eastnorth_to_latlon
from visualization.plot_height_map import plot_height_map
from coordinate_system import global_to_local


def center_crop(nan_mask):
    height, width = nan_mask.shape
    thres = 0.7

    for col_min in range(width):
        empty_ratio = np.sum(nan_mask[:, col_min]) / height
        if empty_ratio < thres:
            break

    for col_max in range(width-1, -1, -1):
        if np.sum(nan_mask[:, col_max]) / height < thres:
            break

    for row_min in range(height):
        if np.sum(nan_mask[row_min, :]) / width < thres:
            break

    for row_max in range(height-1, -1, -1):
        if np.sum(nan_mask[row_max, :]) / width < thres:
            break

    return (col_min, col_max, row_min, row_max)


def process_dsm_gt(work_dir, gt_file):
    gt_subdir = os.path.join(work_dir, 'ground_truth')
    if not os.path.exists(gt_subdir):
        os.mkdir(gt_subdir)

    gt_dsm, meta_dict = read_dsm_tif(gt_file)

    np.save(os.path.join(gt_subdir, 'dsm_gt_data.npy'), gt_dsm)
    with open(os.path.join(gt_subdir, 'dsm_gt_meta.json'), 'w') as fp:
        json.dump(meta_dict, fp, indent=2)

    plot_height_map(gt_dsm, os.path.join(gt_subdir, 'dsm_gt.jpg'), save_cbar=True)

    # generate points
    zz = gt_dsm.reshape((-1, 1))
    valid_mask = np.logical_not(np.isnan(zz))
    zz = zz[valid_mask].reshape((-1, 1))

    # utm points
    xx = np.linspace(meta_dict['ul_northing'], meta_dict['lr_northing'], meta_dict['img_height'])
    yy = np.linspace(meta_dict['ul_easting'], meta_dict['lr_easting'], meta_dict['img_width'])
    xx, yy = np.meshgrid(xx, yy, indexing='ij')     # xx, yy are of shape (height, width)
    xx = xx.reshape((-1, 1))
    yy = yy.reshape((-1, 1))

    xx = xx[valid_mask].reshape((-1, 1))
    yy = yy[valid_mask].reshape((-1, 1))

    # latlon points
    lat, lon = eastnorth_to_latlon(yy, xx, meta_dict['zone_number'], meta_dict['hemisphere'])
    latlonalt_points = np.concatenate((lat, lon, zz), axis=1)
    np.save(os.path.join(gt_subdir, 'dsm_gt_latlonalt_points.npy'), latlonalt_points)

    # bbx latlon
    lat_min = np.min(lat)
    lat_max = np.max(lat)
    lon_min = np.min(lon)
    lon_max = np.max(lon)

    bbx_latlon = {
        'lat_min': lat_min,
        'lat_max': lat_max,
        'lon_min': lon_min,
        'lon_max': lon_max,
        'lat_range': lat_max - lat_min,
        'lon_range': lon_max - lon_min,
        'alt_min': meta_dict['alt_min'],
        'alt_max': meta_dict['alt_max'],
    }
    with open(os.path.join(gt_subdir, 'dsm_gt_bbx_latlonalt.json'), 'w') as fp:
        json.dump(bbx_latlon, fp, indent=2)

    # bbx enu
    xx, yy, zz = global_to_local(work_dir, lat, lon, zz)
    bbx_enu = {
        'e_min': np.min(xx),
        'e_max': np.max(xx),
        'n_min': np.min(yy),
        'n_max': np.max(yy),
        'u_min': np.min(zz),
        'u_max': np.max(zz)
    }

    np.save(os.path.join(gt_subdir, 'dsm_gt_enu_points.npy'), np.hstack((xx, yy, zz)))

    with open(os.path.join(gt_subdir, 'dsm_gt_bbx_enu.json'), 'w') as fp:
        json.dump(bbx_enu, fp, indent=2)


if __name__ == '__main__':
    pass
