import numpy as np
import os
from lib.ply_np_converter import np2ply
import json
import logging
from lib.dsm_util import write_dsm_tif
from visualization.plot_height_map import plot_height_map
from visualization.plot_error_map import plot_error_map


def compute_stats(all_dsm):
    # compute stddev

    # compute mad

    # compute num_measurements

    pass


def run_fuse(work_dir):
    geo_grid_npy_dir = os.path.join(work_dir, 'mvs_results/height_maps/geo_grid_npy')
    out_dir = os.path.join(work_dir, 'mvs_results/aggregate_2p5d')

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

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

    # todo: add median filter here

    # write tif
    with open(os.path.join(work_dir, 'ground_truth/dsm_gt_meta.json')) as fp:
        meta_dict = json.load(fp)

    void_ratio = np.sum(np.isnan(all_dsm_mean_no_outliers)) / all_dsm_mean_no_outliers.size
    with open(os.path.join(out_dir, 'aggregate_2p5d_info.txt'), 'w') as fp:
        fp.write('completeness: {} %'.format((1.0-void_ratio) * 100.0))

    write_dsm_tif(all_dsm_mean_no_outliers, meta_dict, os.path.join(out_dir, 'aggregate_2p5d.tif'))
    plot_height_map(all_dsm_mean_no_outliers, os.path.join(out_dir, 'aggregate_2p5d.jpg'),
                    save_cbar=True, force_range=(meta_dict['alt_min'], meta_dict['alt_max']))

    zz = all_dsm_mean_no_outliers.reshape((-1, 1))
    valid_mask = np.logical_not(np.isnan(zz))
    zz = zz[valid_mask].reshape((-1, 1))

    # generate point cloud
    xx = np.linspace(meta_dict['ul_northing'], meta_dict['lr_northing'], meta_dict['img_height'])
    yy = np.linspace(meta_dict['ul_easting'], meta_dict['lr_easting'], meta_dict['img_width'])
    xx, yy = np.meshgrid(xx, yy, indexing='ij')     # xx, yy are of shape (height, width)
    xx = xx.reshape((-1, 1))
    yy = yy.reshape((-1, 1))

    xx = xx[valid_mask].reshape((-1, 1))
    yy = yy[valid_mask].reshape((-1, 1))

    utm_points = np.concatenate((yy, xx, zz), axis=1)
    comment_1 = 'projection: UTM {}{}'.format(meta_dict['zone_number'], meta_dict['hemisphere'])
    comments = [comment_1,]
    np2ply(utm_points, os.path.join(out_dir, 'aggregate_2p5d.ply'), comments)


if __name__ == '__main__':
    work_dir = '/data2/kz298/mvs3dm_result/MasterProvisional2'
    run_fuse(work_dir)
