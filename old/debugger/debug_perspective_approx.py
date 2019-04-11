import os
import json
from lib.rpc_model import RPCModel
import numpy as np
from lib.compose_proj_mat import compose_proj_mat
from lib.save_image_only import save_image_only
from lib.proj_to_utm_grid import proj_to_utm_grid
import logging


def project_all_rpc(tile_dir):
    rpc_models = []
    img_names = []

    metas_subdir = os.path.join(tile_dir, 'metas/')
    for item in sorted(os.listdir(metas_subdir)):
        img_names.append(item[:-5])     # remove '.json'
        with open(os.path.join(metas_subdir, item)) as fp:
            rpc_models.append(RPCModel(json.load(fp)))

    gt_subdir = os.path.join(tile_dir, 'ground_truth/')
    latlon_pts = np.load(os.path.join(gt_subdir, 'dsm_gt_latlon_points.npy'))
    out_subdir = os.path.join(gt_subdir, 'dsm_pixels_rpc')
    if not os.path.exists(out_subdir):
        os.mkdir(out_subdir)

    cnt = len(rpc_models)
    for i in range(cnt):
        print('projection using {}/{} rpc models...'.format(i+1, cnt))
        col, row = rpc_models[i].projection(latlon_pts[:, 0], latlon_pts[:, 1], latlon_pts[:, 2])

        pixels = np.hstack((col[:, np.newaxis], row[:, np.newaxis]))
        np.save(os.path.join(out_subdir, img_names[i] + '.npy'), pixels)


def project_all_perspective(tile_dir, camera_dict_json, approx_subdir):
    utm_pts = np.load(os.path.join(tile_dir, 'ground_truth/dsm_gt_utm_points.npy'))
    # convert to local coordinate frame
    with open(os.path.join(tile_dir, 'aoi.json')) as fp:
        aoi_dict = json.load(fp)
    ll_east = aoi_dict['ul_easting']
    ll_north = aoi_dict['ul_northing'] - aoi_dict['height']
    utm_pts[:, 0] -= ll_east
    utm_pts[:, 1] -= ll_north

    all_ones = np.ones((utm_pts.shape[0], 1))
    utm_pts = np.hstack((utm_pts, all_ones))

    if not os.path.exists(approx_subdir):
        os.mkdir(approx_subdir)

    with open(camera_dict_json) as fp:
        perspective_dict = json.load(fp)

    img_names = sorted(perspective_dict.keys())
    cnt = len(img_names)
    for i in range(cnt):
        logging.info('projection using {}/{} perspective models...'.format(i+1, cnt))
        img_name = img_names[i]
        params = perspective_dict[img_name][2:]
        P = compose_proj_mat(params)
        tmp = np.dot(utm_pts, P.T)
        col = tmp[:, 0:1] / tmp[:, 2:3]
        row = tmp[:, 1:2] / tmp[:, 2:3]
        pixels = np.hstack((col, row))

        np.save(os.path.join(approx_subdir, img_name[:-4]+'.npy'), pixels)      # remove '.png'


def compare_all(tile_dir, approx_subdir, error_subdir):
    utm_pts = np.load(os.path.join(tile_dir, 'ground_truth/dsm_gt_utm_points.npy'))
    # convert to local coordinate frame
    with open(os.path.join(tile_dir, 'aoi.json')) as fp:
        aoi_dict = json.load(fp)
    ll_east = aoi_dict['ul_easting']
    ll_north = aoi_dict['ul_northing'] - aoi_dict['height']
    utm_pts[:, 0] -= ll_east
    utm_pts[:, 1] -= ll_north

    with open(os.path.join(tile_dir, 'ground_truth/dsm_gt_bbx_local.json')) as fp:
        bbx_local = json.load(fp)

    gt_subdir = os.path.join(tile_dir, 'ground_truth/dsm_pixels_rpc')

    if not os.path.exists(error_subdir):
        os.mkdir(error_subdir)

    items = sorted(os.listdir(approx_subdir))
    logging.info('img_name, min, max, median, mean')
    for item in items:
        approx_pixels = np.load(os.path.join(approx_subdir, item))
        gt_pixels = np.load(os.path.join(gt_subdir, item))

        err = np.sqrt(np.sum((approx_pixels - gt_pixels) ** 2, axis=1))
        img_name = item[:-4]+'.png'
        logging.info('{}, {}, {}, {}, {}'.format(img_name, np.min(err), np.max(err),
                                                            np.median(err), np.mean(err)))

        err_pts = np.hstack((utm_pts[:, 0:1], utm_pts[:, 1:2], err[:, np.newaxis]))
        # project to geo-grid
        err_dsm = proj_to_utm_grid(err_pts, bbx_local['ul_easting'], bbx_local['ul_northing'],
                                   bbx_local['resolution'], bbx_local['img_width'], bbx_local['img_height'])

        err_dsm_empty_mask = np.isnan(err_dsm)
        save_image_only(err_dsm_empty_mask.astype(dtype=np.float),
                        os.path.join(error_subdir, '{}.err.empty_mask.jpg'.format(img_name)), save_cbar=False)
        err_dsm[err_dsm_empty_mask] = np.nanmin(err_dsm)
        save_image_only(err_dsm, os.path.join(error_subdir, '{}.err.jpg'.format(img_name)), save_cbar=True)


def debug_perspective(tile_dir):
    project_all_rpc(tile_dir)

    # before ba
    camera_dict_json = os.path.join(tile_dir, 'colmap/sfm_perspective/init_camera_dict.json')
    approx_subdir = os.path.join(tile_dir, 'ground_truth/dsm_pixels_init')
    error_subdir = os.path.join(tile_dir, 'ground_truth/error_init')
    project_all_perspective(tile_dir, camera_dict_json, approx_subdir)
    compare_all(tile_dir, approx_subdir, error_subdir)

    # after ba
    camera_dict_json = os.path.join(tile_dir, 'colmap/sfm_perspective/final_camera_dict.json')
    approx_subdir = os.path.join(tile_dir, 'ground_truth/dsm_pixels_final')
    error_subdir = os.path.join(tile_dir, 'ground_truth/error_final')
    project_all_perspective(tile_dir, camera_dict_json, approx_subdir)
    compare_all(tile_dir, approx_subdir, error_subdir)


if __name__ == '__main__':
    tile_dir = '/data2/kz298/mvs3dm_result/MasterSequesteredPark'
    gt_file = '/bigdata/kz298/dataset/mvs3dm/Challenge_Data_and_Software/Lidar_gt/MasterSequesteredPark.tif'
    # process_dsm_gt(tile_dir, gt_file)
    debug_perspective(tile_dir)
