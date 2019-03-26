import os
import json
from lib.rpc_model import RPCModel
import numpy as np
from lib.compose_proj_mat import compose_proj_mat
import re


def process_dsm_gt():
    gt_file = os.path.join(debug_dir, 'dsm_gt.tif')
    gt_dsm, geo, proj, meta, width, height = read_image(gt_file)
    gt_dsm = gt_dsm[:, :, 0]
    nan_mask = gt_dsm < -9998   # actual nan value is -9999
    gt_dsm[nan_mask] = np.nan

    utm_zone_str = re.findall('WGS 84 / UTM zone [0-9]{1,2}[N,S]{1}', proj)[0]
    hemisphere = utm_zone_str[-1]
    idx = utm_zone_str.rfind(' ')
    zone_number = int(utm_zone_str[idx:-1])

    tif_meta_dict = {'ul_easting': geo[0],
                     'ul_northing': geo[3],
                     'resolution': geo[1],
                     'zone_number': zone_number,
                     'hemisphere': hemisphere,
                     'width': width,
                     'height': height,
                     'dsm_min': np.nanmin(gt_dsm).item(),
                     'dsm_max': np.nanmax(gt_dsm).item()}

    with open(os.path.join(debug_dir, 'dsm_gt_meta.json'), 'w') as fp:
        json.dump(tif_meta_dict, fp)

    # save data to npy
    np.save(os.path.join(debug_dir, 'dsm_gt_data.npy'), gt_dsm)
    np.save(os.path.join(debug_dir, 'dsm_gt_data_valid_mask.npy'), 1.0-np.float32(nan_mask))


def project_all_rpc(tile_dir):
    rpc_models = []
    img_names = []

    metas_subdir = os.path.join(tile_dir, 'metas/')
    for item in sorted(os.listdir(metas_subdir)):
        img_names.append(item[:-5])     # remove '.json'
        with open(os.path.join(metas_subdir, item)) as fp:
            rpc_models.append(RPCModel(json.load(fp)))

    gt_subdir = os.path.join(tile_dir, 'ground_truth/')
    latlon_pts = np.load(os.path.join(gt_subdir, 'dsm_latlon_points.npy'))
    out_subdir = os.path.join(gt_subdir, 'dsm_pixels_rpc')
    if not os.path.exists(out_subdir):
        os.mkdir(out_subdir)

    cnt = len(rpc_models)
    for i in range(cnt):
        print('projection using {}/{} rpc models...'.format(i+1, cnt))
        col, row = rpc_models[i].projection(latlon_pts[:, 0], latlon_pts[:, 1], latlon_pts[:, 2])

        pixels = np.hstack((col[:, np.newaxis], row[:, np.newaxis]))
        np.save(os.path.join(out_subdir, img_names[i] + '.npy'), pixels)


def project_all_perspective(tile_dir):
    utm_pts = np.load(os.path.join(tile_dir, 'ground_truth/dsm_utm_points.npy'))
    # convert to local coordinate frame
    with open(os.path.join(tile_dir, 'aoi.json')) as fp:
        aoi_dict = json.load(fp)
    ll_east = aoi_dict['ul_easting']
    ll_north = aoi_dict['ul_northing'] - aoi_dict['height']
    utm_pts[:, 0] -= ll_east
    utm_pts[:, 1] -= ll_north

    all_ones = np.ones((utm_pts.shape[0], 1))
    utm_pts = np.hstack((utm_pts, all_ones))

    approx_subdir = os.path.join(tile_dir, 'ground_truth/dsm_pixels_approx')

    if not os.path.exists(approx_subdir):
        os.mkdir(approx_subdir)

    with open(os.path.join(tile_dir, 'approx_perspective_utm.json')) as fp:
        perspective_dict = json.load(fp)

    img_names = sorted(perspective_dict.keys())
    cnt = len(img_names)
    for i in range(cnt):
        print('projection using {}/{} perspective models...'.format(i+1, cnt))
        img_name = img_names[i]
        params = perspective_dict[img_name][2:]
        P = compose_proj_mat(params)
        tmp = np.dot(utm_pts, P.T)
        col = tmp[:, 0:1] / tmp[:, 2:3]
        row = tmp[:, 1:2] / tmp[:, 2:3]
        pixels = np.hstack((col, row))

        np.save(os.path.join(approx_subdir, img_name[:-4]+'.npy'), pixels)      # remove '.png'


from lib.image_util import read_image
from lib.dsm_merger_offline import DsmMerger
from lib.save_image_only import save_image_only


def compare_all(tile_dir):
    utm_pts = np.load(os.path.join(tile_dir, 'ground_truth/dsm_utm_points.npy'))
    # convert to local coordinate frame
    with open(os.path.join(tile_dir, 'aoi.json')) as fp:
        aoi_dict = json.load(fp)
    ll_east = aoi_dict['ul_easting']
    ll_north = aoi_dict['ul_northing'] - aoi_dict['height']
    utm_pts[:, 0] -= ll_east
    utm_pts[:, 1] -= ll_north

    with open(os.path.join(tile_dir, 'ground_truth/bbx_utm.json')) as fp:
        bbx_utm = json.load(fp)

    bbx = (bbx_utm['ul_easting'] - ll_east, bbx_utm['ul_northing'] - ll_north, bbx_utm['img_width'], bbx_utm['img_height'])
    resolution = 0.3
    dsm_merger = DsmMerger(bbx, resolution)

    gt_subdir = os.path.join(tile_dir, 'ground_truth/dsm_pixels_rpc')
    approx_subdir = os.path.join(tile_dir, 'ground_truth/dsm_pixels_approx')
    out_dir = os.path.join(tile_dir, 'ground_truth/error')
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    items = sorted(os.listdir(approx_subdir))
    print('img_name, min, max, median, mean')
    for item in items:
        approx_pixels = np.load(os.path.join(approx_subdir, item))
        gt_pixels = np.load(os.path.join(gt_subdir, item))

        err = np.sqrt(np.sum((approx_pixels - gt_pixels) ** 2, axis=1))
        img_name = item[:-4]+'.png'
        print('{}, {}, {}, {}, {}'.format(img_name, np.min(err), np.max(err),
                                                            np.median(err), np.mean(err)))

        err_pts = np.hstack((utm_pts[:, 0:1], utm_pts[:, 1:2], err[:, np.newaxis]))
        # project to geo-grid
        err_dsm = dsm_merger.add(err_pts)

        err_dsm_empty_mask = np.isnan(err_dsm)
        save_image_only(err_dsm_empty_mask.astype(dtype=np.float),
                        os.path.join(out_dir, '{}.err.empty_mask.jpg'.format(img_name)), save_cbar=False)
        err_dsm[err_dsm_empty_mask] = np.nanmin(err_dsm)
        save_image_only(err_dsm, os.path.join(out_dir, '{}.err.jpg'.format(img_name)), save_cbar=True)

    err_dsm = dsm_merger.get_merged_dsm_mean()
    err_dsm_empty_mask = np.isnan(err_dsm)
    save_image_only(err_dsm_empty_mask.astype(dtype=np.float),
                    os.path.join(out_dir, 'mean_err.empty_mask.jpg'), save_cbar=False)
    err_dsm[err_dsm_empty_mask] = 0.
    save_image_only(err_dsm, os.path.join(out_dir, 'mean_err.dsm.jpg'), save_cbar=True)


if __name__ == '__main__':
    tile_dir = '/data2/kz298/mvs3dm_result/MasterSequesteredPark'
    #project_all_rpc(tile_dir)
    #
    #project_all_perspective(tile_dir)

    compare_all(tile_dir)
