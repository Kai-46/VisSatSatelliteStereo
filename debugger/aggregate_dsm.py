import numpy as np
from lib.save_image_only import save_image_only
import os
from lib.ply_np_converter import np2ply
import json


def merge_dsm(tile_dir):
    geo_grid_npy_dir = os.path.join(tile_dir, 'mvs_results/height_maps/geo_grid_npy')
    out_dir = os.path.join(tile_dir, 'mvs_results/')

    all_dsm = []
    for item in os.listdir(geo_grid_npy_dir):
        dsm = np.load(os.path.join(geo_grid_npy_dir, item))
        print('current dsm empty ratio: {} %'.format(np.sum(np.isnan(dsm)) / dsm.size))

        all_dsm.append(dsm[:, :, np.newaxis])

    all_dsm = np.concatenate(all_dsm, axis=2)

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


def save_stats(stats, name, out_dir):
    np.save(os.path.join(out_dir, '{}.npy'.format(name)), stats)

    nan_mask = np.isnan(stats)
    stats[nan_mask] = np.nanmin(stats)
    save_image_only(stats, os.path.join(out_dir, '{}.jpg'.format(name)), save_cbar=True)
    save_image_only(1.0-np.float32(nan_mask), os.path.join(out_dir, '{}.mask.jpg'.format(name)), plot=False)


def aggregate_dsm(geo_grid_npy_dir, out_dir, tile_dir):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    all_dsm = []
    for item in os.listdir(geo_grid_npy_dir):
        dsm = np.load(os.path.join(geo_grid_npy_dir, item))
        print('current dsm empty ratio: {} %'.format(np.sum(np.isnan(dsm)) / dsm.size))

        all_dsm.append(dsm[:, :, np.newaxis])

    all_dsm = np.concatenate(all_dsm, axis=2)

    # output max
    all_dsm_max = np.nanmax(all_dsm, axis=2)
    save_stats(all_dsm_max, 'dsm_max', out_dir)

    # output mean
    all_dsm_mean = np.nanmean(all_dsm, axis=2)
    save_stats(all_dsm_mean, 'dsm_mean', out_dir)

    # check stddev
    all_dsm_stddev = np.nanstd(all_dsm, axis=2)
    # clip
    all_dsm_stddev[all_dsm_stddev > 1.0] = 1.0
    save_stats(all_dsm_stddev, 'dsm_stddev', out_dir)

    # output median
    all_dsm_median = np.nanmedian(all_dsm, axis=2)
    all_dsm_median_valid_mask = np.logical_not(np.isnan(all_dsm_median))
    save_stats(all_dsm_median, 'dsm_median', out_dir)

    # check mad
    all_dsm_mad = np.nanmedian(np.abs(all_dsm - all_dsm_median[:, :, np.newaxis]), axis=2)

    # output mean after removing outliers
    outlier_mask = np.abs(all_dsm - all_dsm_median[:, :, np.newaxis]) > all_dsm_mad[:, :, np.newaxis]
    all_dsm[outlier_mask] = np.nan

    # output mean
    all_dsm_mean_no_outliers = np.nanmean(all_dsm, axis=2)
    save_stats(all_dsm_mean_no_outliers, 'dsm_mean_no_outliers', out_dir)

    all_dsm_mad[all_dsm_mad > 1.0] = 1.0
    save_stats(all_dsm_mad, 'dsm_mad', out_dir)

    dsm_gt = np.load(os.path.join(tile_dir, 'ground_truth/dsm_gt_data.npy'))
    dsm_gt_valid_mask = np.load(os.path.join(tile_dir, 'ground_truth/dsm_gt_data_valid_mask.npy'))
    signed_err = all_dsm_mean_no_outliers - dsm_gt
    err = np.abs(signed_err)

    median_err = np.median(err[np.logical_and(dsm_gt_valid_mask, all_dsm_median_valid_mask)])

    completeness = np.sum(np.float32(err < 1.0) * dsm_gt_valid_mask) / dsm_gt_valid_mask.size

    print('{}, median_err: {}, completeness: {}'.format(geo_grid_npy_dir, median_err, completeness))

    from debugger.signed_colormap import get_signed_colormap
    np.save(os.path.join(out_dir, 'error.npy'), signed_err)

    signed_err[signed_err < -2.0] = -2.0
    signed_err[signed_err > 2.0] = 2.0
    signed_err[np.logical_not(dsm_gt_valid_mask)] = 0.0
    cmap, norm = get_signed_colormap()

    nan_mask = np.isnan(signed_err)
    signed_err[nan_mask] = np.nanmin(signed_err)
    save_image_only(signed_err, os.path.join(out_dir, 'error.jpg'), save_cbar=True, cmap=cmap, norm=norm)
    save_image_only(1.0-np.float32(nan_mask), os.path.join(out_dir, 'error.mask.jpg'), plot=False)


def compute_err(tile_dir):
    pass


if __name__ == '__main__':
    tile_dir = '/data2/kz298/mvs3dm_result/MasterSequesteredPark'
    for win_rad in [2, 3, 5, 7, 9, 11, 13]:
        geo_grid_npy_dir = os.path.join(tile_dir, 'mvs_results_winrad_{}/height_maps/geo_grid_npy'.format(win_rad))
        out_dir = os.path.join(tile_dir, 'mvs_results_winrad_{}/height_maps/geo_grid_aggregate'.format(win_rad))
        print('{}'.format(geo_grid_npy_dir))
        aggregate_dsm(geo_grid_npy_dir, out_dir, tile_dir)
