import os
import json
from lib.rpc_model import RPCModel
from lib.gen_grid import gen_grid
from lib.solve_affine import solve_affine
from lib.solve_perspective import solve_perspective
import numpy as np
from pyquaternion import Quaternion
from lib.check_error import check_perspective_error
import logging
from lib.latlon_utm_converter import eastnorth_to_latlon
from coordinate_system import global_to_local


def discretize_volume(work_dir):
    bbx_file = os.path.join(work_dir, 'aoi.json')
    with open(bbx_file) as fp:
        bbx = json.load(fp)

    ul_easting = bbx['ul_easting']
    ul_northing = bbx['ul_northing']
    lr_easting = bbx['lr_easting']
    lr_northing = bbx['lr_northing']
    zone_number = bbx['zone_number']
    hemisphere = bbx['hemisphere']
    alt_min = bbx['alt_min']
    alt_max = bbx['alt_max']

    # each grid-cell is about 5 meters * 5 meters * 5 meters
    xy_axis_grid_points = 100
    z_axis_grid_points = 20

    # create north_east_height grid
    # note that this is a left-handed coordinate system
    north_points = np.linspace(ul_northing, lr_northing, xy_axis_grid_points)
    east_points = np.linspace(ul_easting, lr_easting, xy_axis_grid_points)
    alt_points = np.linspace(alt_min, alt_max, z_axis_grid_points)
    north_points, east_points, alt_points = gen_grid(north_points, east_points, alt_points)

    # convert to lat lon
    lat_points, lon_points = eastnorth_to_latlon(east_points, north_points, zone_number, hemisphere)

    # convert to local utm
    ll_easting = ul_easting
    ll_northing = lr_northing
    xx_utm = east_points - ll_easting
    yy_utm = north_points - ll_northing
    zz_utm = alt_points

    # convert to enu
    latlonalt = np.hstack((lat_points, lon_points, alt_points))
    utm_local = np.hstack((xx_utm, yy_utm, zz_utm))
    xx_enu, yy_enu, zz_enu = global_to_local(work_dir, lat_points, lon_points, alt_points)
    enu = np.hstack((xx_enu, yy_enu, zz_enu))
    return latlonalt, utm_local, enu


class CameraApprox(object):
    def __init__(self, work_dir):
        self.work_dir = work_dir

        self.latlonalt, self.utm_local, self.enu = discretize_volume(work_dir)

        self.img_names = []
        self.rpc_models = []
        self.region_dicts = []

        metas_subdir = os.path.join(self.work_dir, 'metas/')
        for item in sorted(os.listdir(metas_subdir)):
            self.img_names.append(item[:-5] + '.png')
            with open(os.path.join(metas_subdir, item)) as fp:
                self.rpc_models.append(RPCModel(json.load(fp)))

        self.cnt = len(self.rpc_models)

        self.out_dir = os.path.join(work_dir, 'approx_camera')
        if not os.path.exists(self.out_dir):
            os.mkdir(self.out_dir)

    def approx_affine_latlonalt(self):
        logging.info('deriving an affine camera approximation...')
        logging.info('scene coordinate frame is in lat, lon, alt')

        lat_points = self.latlonalt[:, 0:1]
        lon_points = self.latlonalt[:, 1:2]
        alt_points = self.latlonalt[:, 2:3]

        affine_dict = {}
        for i in range(self.cnt):
            col, row = self.rpc_models[i].projection(lat_points, lon_points, alt_points)

            # make sure all the points lie inside the image
            width = self.rpc_models[i].width
            height = self.rpc_models[i].height
            keep_mask = np.logical_and(col >= 0, row >= 0)
            keep_mask = np.logical_and(keep_mask, col < width)
            keep_mask = np.logical_and(keep_mask, row < height)

            P = solve_affine(lat_points, lon_points, alt_points, col, row, keep_mask)

            # write to file
            img_name = self.img_names[i]
            P = list(P.reshape((8,)))
            affine_dict[img_name] = [width, height] + P

        with open(os.path.join(self.out_dir, 'affine_latlonalt.json'), 'w') as fp:
            json.dump(affine_dict, fp, indent=2)

        bbx = { 'lat_min': np.min(lat_points),
                'lat_max': np.max(lat_points),
                'lon_min': np.min(lon_points),
                'lon_max': np.max(lon_points),
                'alt_min': np.min(alt_points),
                'alt_max': np.max(alt_points)}
        with open(os.path.join(self.out_dir, 'bbx_latlonalt.json'), 'w') as fp:
            json.dump(bbx, fp, indent=2)

    def approx_perspective_utmalt(self):
        logging.info('deriving a perspective camera approximation...')
        logging.info('scene coordinate frame is in local UTM, alt')

        perspective_dict = {}

        errors_txt = 'img_name, mean_proj_err (pixels), median_proj_err (pixels), max_proj_err (pixels), mean_inv_proj_err (meters), median_inv_proj_err (meters), max_inv_proj_err (meters)\n'

        lat_points = self.latlonalt[:, 0:1]
        lon_points = self.latlonalt[:, 1:2]
        alt_points = self.latlonalt[:, 2:3]

        xx = self.utm_local[:, 0:1]
        yy = self.utm_local[:, 1:2]
        zz = self.utm_local[:, 2:3]

        for i in range(self.cnt):

            col, row = self.rpc_models[i].projection(lat_points, lon_points, alt_points)

            # make sure all the points lie inside the image
            width = self.rpc_models[i].width
            height = self.rpc_models[i].height
            keep_mask = np.logical_and(col >= 0, row >= 0)
            keep_mask = np.logical_and(keep_mask, col < width)
            keep_mask = np.logical_and(keep_mask, row < height)

            K, R, t = solve_perspective(xx, yy, zz, col, row, keep_mask)

            qvec = Quaternion(matrix=R)
            # fx, fy, cx, cy, s, qvec, t
            params = [width, height, K[0, 0], K[1, 1], K[0, 2], K[1, 2], K[0, 1],
                       qvec[0], qvec[1], qvec[2], qvec[3],
                       t[0, 0], t[1, 0], t[2, 0]]

            img_name = self.img_names[i]
            perspective_dict[img_name] = params

            # check approximation error
            tmp = check_perspective_error(xx, yy, zz, col, row, K, R, t, keep_mask)
            errors_txt += '{}, {}, {}, {}, {}, {}, {}\n'.format(img_name, tmp[0], tmp[1], tmp[2], tmp[3], tmp[4], tmp[5])

        with open(os.path.join(self.out_dir, 'perspective_utmalt.json'), 'w') as fp:
            json.dump(perspective_dict, fp, indent=2)

        with open(os.path.join(self.out_dir, 'perspective_utmalt_error.csv'), 'w') as fp:
            fp.write(errors_txt)

        bbx = { 'xx_min': np.min(xx),
                'xx_max': np.max(xx),
                'yy_min': np.min(yy),
                'yy_max': np.max(yy),
                'zz_min': np.min(zz),
                'zz_max': np.max(zz)}
        with open(os.path.join(self.out_dir, 'bbx_utmalt.json'), 'w') as fp:
            json.dump(bbx, fp, indent=2)

    def approx_perspective_enu(self):
        logging.info('deriving a perspective camera approximation...')
        logging.info('scene coordinate frame is in ENU')

        perspective_dict = {}

        errors_txt = 'img_name, mean_proj_err (pixels), median_proj_err (pixels), max_proj_err (pixels), mean_inv_proj_err (meters), median_inv_proj_err (meters), max_inv_proj_err (meters)\n'

        lat_points = self.latlonalt[:, 0:1]
        lon_points = self.latlonalt[:, 1:2]
        alt_points = self.latlonalt[:, 2:3]

        xx = self.enu[:, 0:1]
        yy = self.enu[:, 1:2]
        zz = self.enu[:, 2:3]

        for i in range(self.cnt):
            col, row = self.rpc_models[i].projection(lat_points, lon_points, alt_points)

            # make sure all the points lie inside the image
            width = self.rpc_models[i].width
            height = self.rpc_models[i].height
            keep_mask = np.logical_and(col >= 0, row >= 0)
            keep_mask = np.logical_and(keep_mask, col < width)
            keep_mask = np.logical_and(keep_mask, row < height)

            K, R, t = solve_perspective(xx, yy, zz, col, row, keep_mask)

            qvec = Quaternion(matrix=R)
            # fx, fy, cx, cy, s, qvec, t
            params = [width, height, K[0, 0], K[1, 1], K[0, 2], K[1, 2], K[0, 1],
                       qvec[0], qvec[1], qvec[2], qvec[3],
                       t[0, 0], t[1, 0], t[2, 0]]

            img_name = self.img_names[i]
            perspective_dict[img_name] = params

            # check approximation error
            tmp = check_perspective_error(xx, yy, zz, col, row, K, R, t, keep_mask)
            errors_txt += '{}, {}, {}, {}, {}, {}, {}\n'.format(img_name, tmp[0], tmp[1], tmp[2], tmp[3], tmp[4], tmp[5])

        with open(os.path.join(self.out_dir, 'perspective_enu.json'), 'w') as fp:
            json.dump(perspective_dict, fp, indent=2)

        with open(os.path.join(self.out_dir, 'perspective_enu_error.csv'), 'w') as fp:
            fp.write(errors_txt)

        bbx = { 'xx_min': np.min(xx),
                'xx_max': np.max(xx),
                'yy_min': np.min(yy),
                'yy_max': np.max(yy),
                'zz_min': np.min(zz),
                'zz_max': np.max(zz)}
        with open(os.path.join(self.out_dir, 'bbx_enu.json'), 'w') as fp:
            json.dump(bbx, fp, indent=2)


if __name__ == '__main__':
    pass
