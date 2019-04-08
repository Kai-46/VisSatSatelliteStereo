import numpy as np
import os
from lib.ply_np_converter import np2ply
import json
from debugger.save_stats import save_stats
import logging


def merge_dsm(tile_dir):
    geo_grid_npy_dir = os.path.join(tile_dir, 'mvs_results/height_maps/geo_grid_npy')
    out_dir = os.path.join(tile_dir, 'mvs_results/')

    all_dsm = []
    for item in os.listdir(geo_grid_npy_dir):
        dsm = np.load(os.path.join(geo_grid_npy_dir, item))
        logging.info('current dsm empty ratio: {} %'.format(np.sum(np.isnan(dsm)) / dsm.size))

        all_dsm.append(dsm[:, :, np.newaxis])

    cnt = len(all_dsm)
    all_dsm = np.concatenate(all_dsm, axis=2)

    # reject two measurements
    num_measurements = cnt - np.sum(np.isnan(all_dsm), axis=2, keepdims=True)
    mask = np.tile(num_measurements <= 2, (1, 1, cnt))
    all_dsm[mask] = np.nan

    all_dsm_median = np.nanmedian(all_dsm, axis=2, keepdims=True)
    all_dsm_mad = np.nanmedian(np.abs(all_dsm - all_dsm_median), axis=2, keepdims=True)
    # output mean after removing outliers
    outlier_mask = np.abs(all_dsm - all_dsm_median) > all_dsm_mad
    all_dsm[outlier_mask] = np.nan

    all_dsm_mean_no_outliers = np.nanmean(all_dsm, axis=2)

    # generate point cloud
    with open(os.path.join(tile_dir, 'ground_truth/dsm_gt_bbx_utm.json')) as fp:
        bbx_utm = json.load(fp)

    zz = all_dsm_mean_no_outliers.reshape((-1, 1))
    valid_mask = np.logical_not(np.isnan(zz))
    zz = zz[valid_mask].reshape((-1, 1))

    # utm points
    xx = np.linspace(bbx_utm['ul_northing'], bbx_utm['lr_northing'], bbx_utm['img_height'])
    yy = np.linspace(bbx_utm['ul_easting'], bbx_utm['lr_easting'], bbx_utm['img_width'])
    xx, yy = np.meshgrid(xx, yy, indexing='ij')     # xx, yy are of shape (height, width)
    xx = xx.reshape((-1, 1))
    yy = yy.reshape((-1, 1))

    xx = xx[valid_mask].reshape((-1, 1))
    yy = yy[valid_mask].reshape((-1, 1))

    utm_points = np.concatenate((yy, xx, zz), axis=1)
    comment_1 = 'projection: UTM {}{}'.format(bbx_utm['zone_number'], bbx_utm['hemisphere'])
    comments = [comment_1,]
    np2ply(utm_points, os.path.join(out_dir, 'merged_dsm.ply'), comments)

    save_stats(all_dsm_mean_no_outliers, 'merged_dsm', out_dir)
