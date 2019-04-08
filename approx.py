import os
import json
from lib.rpc_model import RPCModel
from lib.gen_grid import gen_grid
from lib.solve_affine import solve_affine
from lib.solve_perspective import solve_perspective
import utm
import numpy as np
from pyquaternion import Quaternion
from lib.check_error import check_perspective_error
import logging
from lib.latlon_utm_converter import eastnorth_to_latlon


class Approx(object):
    def __init__(self, tile_dir):
        self.tile_dir = tile_dir

        # needs to use a common world coordinate frame for all smaller regions
        # common coordinate frame is:
        #   aoi lower-left easting, aoi lower-left northing, aoi lower-left height
        with open(os.path.join(tile_dir, 'aoi.json')) as fp:
            self.aoi_dict = json.load(fp)

        self.img_names = []
        self.rpc_models = []
        self.region_dicts = []

        metas_subdir = os.path.join(self.tile_dir, 'metas/')
        regions_subdir = os.path.join(self.tile_dir, 'regions/')
        for item in sorted(os.listdir(metas_subdir)):
            self.img_names.append(item[:-5] + '.png')
            with open(os.path.join(metas_subdir, item)) as fp:
                self.rpc_models.append(RPCModel(json.load(fp)))
            with open(os.path.join(regions_subdir, item)) as fp:
                self.region_dicts.append(json.load(fp))

        self.cnt = len(self.rpc_models)

    # def approx(self):
    #     self.approx_affine_latlon()
    #     self.approx_perspective_utm()

    def approx_affine_latlon(self):
        logging.info('deriving an affine camera approximation...')
        logging.info('scene coordinate frame is in latitude, longitude')

        affine_dict = {}
        for i in range(self.cnt):
            region_dict = self.region_dicts[i]
            ul_east = region_dict['ul_easting']
            ul_north = region_dict['ul_northing']
            lr_east = ul_east + region_dict['width']
            lr_north = ul_north - region_dict['height']
            zone_number = region_dict['zone_number']
            hemisphere = region_dict['hemisphere']
            min_height = region_dict['min_z']
            max_height = region_dict['max_z']

            logging.info('height min, max: {}, {}'.format(min_height, max_height))

            northern = True if hemisphere == 'N' else False

            # convert to lat, lon
            ul_lat, ul_lon = utm.to_latlon(ul_east, ul_north, zone_number, northern=northern)
            lr_lat, lr_lon = utm.to_latlon(lr_east, lr_north, zone_number, northern=northern)

            # create grid
            xy_axis_grid_points = 100
            z_axis_grid_points = 20
            lat_points = np.linspace(ul_lat, lr_lat, xy_axis_grid_points)
            lon_points = np.linspace(ul_lon, lr_lon, xy_axis_grid_points)
            z_points = np.linspace(min_height, max_height, z_axis_grid_points)

            lat_points, lon_points, z_points = gen_grid(lat_points, lon_points, z_points)

            col, row = self.rpc_models[i].projection(lat_points, lon_points, z_points)

            # make sure all the points lie inside the image
            width = self.rpc_models[i].width
            height = self.rpc_models[i].height
            keep_mask = np.logical_and(col >= 0, row >= 0)
            keep_mask = np.logical_and(keep_mask, col < width)
            keep_mask = np.logical_and(keep_mask, row < height)

            P = solve_affine(lat_points, lon_points, z_points, col, row, keep_mask)

            # write to file
            img_name = self.img_names[i]
            P = list(P.reshape((8,)))
            affine_dict[img_name] = [width, height] + P

        return affine_dict

    def approx_perspective_utm(self):
        logging.info('deriving a perspective camera approximation...')
        logging.info('scene coordinate frame is in UTM')

        perspective_dict = {}

        errors_txt = '\nimg_name, mean_proj_err (pixels), median_proj_err (pixels), max_proj_err (pixels), mean_inv_proj_err (meters), median_inv_proj_err (meters), max_inv_proj_err (meters)\n'

        aoi_ll_east = self.aoi_dict['ul_easting']
        aoi_ll_north = self.aoi_dict['ul_northing'] - self.aoi_dict['height']

        for i in range(self.cnt):
            region_dict = self.region_dicts[i]
            ul_east = region_dict['ul_easting']
            ul_north = region_dict['ul_northing']
            lr_east = ul_east + region_dict['width']
            lr_north = ul_north - region_dict['height']
            zone_number = region_dict['zone_number']
            hemisphere = region_dict['hemisphere']
            min_height = region_dict['min_z']
            max_height = region_dict['max_z']

            logging.info('height min, max: {}, {}'.format(min_height, max_height))

            # each grid-cell is about 5 meters * 5 meters * 5 meters
            xy_axis_grid_points = 100
            z_axis_grid_points = 20

            # create north_east_height grid
            # note that this is a left-handed coordinate system
            north_points = np.linspace(ul_north, lr_north, xy_axis_grid_points)
            east_points = np.linspace(ul_east, lr_east, xy_axis_grid_points)
            z_points = np.linspace(min_height, max_height, z_axis_grid_points)
            north_points, east_points, z_points = gen_grid(north_points, east_points, z_points)

            lat_points, lon_points = eastnorth_to_latlon(east_points, north_points, zone_number, hemisphere)

            # convert to lat, lon
            col, row = self.rpc_models[i].projection(lat_points, lon_points, z_points)

            # change to the right-handed coordinate frame and use a smaller number
            xx = east_points - aoi_ll_east
            yy = north_points - aoi_ll_north
            zz = z_points

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
            errors_txt += '{}: {}, {}, {}, {}, {}, {}\n'.format(img_name, tmp[0], tmp[1], tmp[2], tmp[3], tmp[4], tmp[5])

        logging.info(errors_txt)

        return perspective_dict


if __name__ == '__main__':
    work_dir = '/data2/kz298/core3d_aoi/aoi-d4-jacksonville'
    appr = Approx(work_dir)
    perspective_dict = appr.approx_perspective_utm()
    with open(os.path.join(work_dir, 'approx_perspective_utm.json'), 'w') as fp:
        json.dump(perspective_dict, fp, indent=2)

    affine_dict = appr.approx_affine_latlon()
    with open(os.path.join(work_dir, 'approx_affine_latlon.json'), 'w') as fp:
        json.dump(affine_dict, fp, indent=2)
