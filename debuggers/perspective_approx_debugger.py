import os
import json
from lib.rpc_model import RPCModel
import numpy as np
from lib.compose_proj_mat import compose_proj_mat
from lib.proj_to_utm_grid import proj_to_utm_grid
import logging
from visualization.plot_error_map import plot_error_map


def project_all_rpc(work_dir):
    rpc_models = []
    img_names = []

    with open(os.path.join(work_dir, 'metas.json')) as fp:
        metas = json.load(fp)
    for img_name in sorted(metas.keys()):
        img_names.append(img_name)
        rpc_models.append(RPCModel(metas[img_name]))

    gt_subdir = os.path.join(work_dir, 'ground_truth/')
    latlonalt_pts = np.load(os.path.join(gt_subdir, 'dsm_gt_latlonalt_points.npy'))
    out_subdir = os.path.join(gt_subdir, 'dsm_pixels_rpc')
    if not os.path.exists(out_subdir):
        os.mkdir(out_subdir)

    cnt = len(rpc_models)
    for i in range(cnt):
        logging.info('projection using {}/{} rpc models...'.format(i+1, cnt))
        col, row = rpc_models[i].projection(latlonalt_pts[:, 0:1], latlonalt_pts[:, 1:2], latlonalt_pts[:, 2:3])

        pixels = np.hstack((col, row))
        img_name = img_names[i]
        np.save(os.path.join(out_subdir, img_name[:-4] + '.npy'), pixels) # remove '.png'


def project_all_perspective(work_dir, camera_dict_json, approx_subdir):
    gt_subdir = os.path.join(work_dir, 'ground_truth/')
    pts = np.load(os.path.join(gt_subdir, 'dsm_gt_enu_points.npy'))

    all_ones = np.ones((pts.shape[0], 1))
    pts = np.hstack((pts, all_ones))

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
        tmp = np.dot(pts, P.T)
        col = tmp[:, 0:1] / tmp[:, 2:3]
        row = tmp[:, 1:2] / tmp[:, 2:3]
        pixels = np.hstack((col, row))

        np.save(os.path.join(approx_subdir, img_name[:-4]+'.npy'), pixels)  # remove '.png'


def compare_all(work_dir, approx_subdir, error_subdir):
    gt_subdir = os.path.join(work_dir, 'ground_truth/')
    latlonalt_pts = np.load(os.path.join(gt_subdir, 'dsm_gt_latlonalt_points.npy'))

    with open(os.path.join(gt_subdir, 'dsm_gt_meta.json')) as fp:
        meta_dict = json.load(fp)

    rpc_pixels_subdir = os.path.join(work_dir, 'ground_truth/dsm_pixels_rpc')

    if not os.path.exists(error_subdir):
        os.mkdir(error_subdir)

    items = sorted(os.listdir(approx_subdir))
    logging.info('img_name, min, max, median, mean')
    for item in items:
        approx_pixels = np.load(os.path.join(approx_subdir, item))
        gt_pixels = np.load(os.path.join(rpc_pixels_subdir, item))

        err = np.sqrt(np.sum((approx_pixels - gt_pixels) ** 2, axis=1))
        img_name = item[:-4]+'.png'
        logging.info('{}, {}, {}, {}, {}'.format(img_name, np.min(err), np.max(err),
                                                            np.median(err), np.mean(err)))

        err_pts = np.hstack((latlonalt_pts[:, 0:1], latlonalt_pts[:, 1:2], err[:, np.newaxis]))
        # project to geo-grid
        err_dsm = proj_to_utm_grid(err_pts, meta_dict['ul_easting'], meta_dict['ul_northing'],
                                   meta_dict['east_resolution'], meta_dict['north_resolution'],
                                   meta_dict['img_width'], meta_dict['img_height'])
        plot_error_map(err_dsm, os.path.join(error_subdir, '{}.error.jpg'.format(img_name)))


def debug_approx(work_dir):
    project_all_rpc(work_dir)

    # before ba
    camera_dict_json = os.path.join(work_dir, 'colmap/sfm_perspective/init_camera_dict.json')
    approx_subdir = os.path.join(work_dir, 'ground_truth/dsm_pixels_init')
    error_subdir = os.path.join(work_dir, 'ground_truth/error_init')
    project_all_perspective(work_dir, camera_dict_json, approx_subdir)
    compare_all(work_dir, approx_subdir, error_subdir)

    # after ba
    camera_dict_json = os.path.join(work_dir, 'colmap/sfm_perspective/init_ba_camera_dict.json')
    approx_subdir = os.path.join(work_dir, 'ground_truth/dsm_pixels_init_ba')
    error_subdir = os.path.join(work_dir, 'ground_truth/error_init_ba')
    project_all_perspective(work_dir, camera_dict_json, approx_subdir)
    compare_all(work_dir, approx_subdir, error_subdir)


if __name__ == '__main__':
    work_dir = '/data2/kz298/mvs3dm_result/MasterProvisional2'
    debug_approx(work_dir)
