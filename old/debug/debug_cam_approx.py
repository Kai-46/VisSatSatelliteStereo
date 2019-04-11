from old.image_util import read_image
import os
import json
import re
import numpy as np
import utm
from lib.parallel_apply_along_axis import parallel_apply_along_axis


debug_dir = '/data2/kz298/mvs3dm_result/debug/cam_approx'


def map_to_utm(latlonalt):
    tmp = utm.from_latlon(latlonalt[0], latlonalt[1])

    return np.array([tmp[0], tmp[1], latlonalt[2]])


def latlonalt_to_eastnorthalt(lat, lon, alt):
    latlonalt = np.concatenate((lat, lon, alt), axis=1)

    # tmp = np.apply_along_axis(map_to_utm, 1, latlonalt)
    tmp = parallel_apply_along_axis(map_to_utm, 1, latlonalt)
    east = tmp[:, 0:1]
    north = tmp[:, 1:2]

    # cnt = lat.shape[0]
    # east = np.zeros((cnt, 1))
    # north = np.zeros((cnt, 1))
    # for i in range(cnt):
    #     tmp = utm.from_latlon(lat[i, 0], lon[i, 0])
    #     east[i, 0] = tmp[0]
    #     north[i, 0] = tmp[1]

    return east, north, alt


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


# bounding box: utm, latlon, local
def gen_bbx():
    with open(os.path.join(debug_dir, 'dsm_gt_meta.json')) as fp:
        tif_meta_dict = json.load(fp)

    ul_easting = tif_meta_dict['ul_easting']
    ul_northing = tif_meta_dict['ul_northing']
    resolution = tif_meta_dict['resolution']
    zone_number = tif_meta_dict['zone_number']
    hemisphere = tif_meta_dict['hemisphere']
    width = tif_meta_dict['width']
    height = tif_meta_dict['height']
    z_min = tif_meta_dict['dsm_min']
    z_max = tif_meta_dict['dsm_max']

    # utm bbx
    lr_easting = ul_easting + resolution * width
    lr_northing = ul_northing - resolution * height

    bbx_utm = {
        'ul_easting': ul_easting,
        'ul_northing': ul_northing,
        'lr_easting': lr_easting,
        'lr_northing': lr_northing,
        'z_min': z_min,
        'z_max': z_max,
        'zone_number': zone_number,
        'hemisphere': hemisphere,
        'img_width': width,
        'img_height': height
    }
    with open(os.path.join(debug_dir, 'bbx_utm.json'), 'w') as fp:
        json.dump(bbx_utm, fp)

    # latlon bbx
    northern = True if hemisphere == 'N' else False
    ul_lat, ul_lon = utm.to_latlon(ul_easting, ul_northing, zone_number, northern=northern)
    lr_lat, lr_lon = utm.to_latlon(lr_easting, lr_northing, zone_number, northern=northern)

    bbx_latlon = {
        'ul_lat': ul_lat,
        'ul_lon': ul_lon,
        'lr_lat': lr_lat,
        'lr_lon': lr_lon,
        'z_min': z_min,
        'z_max': z_max,
        'img_width': width,
        'img_height': height
    }

    with open(os.path.join(debug_dir, 'bbx_latlon.json'), 'w') as fp:
        json.dump(bbx_latlon, fp)


from lib.gen_grid import gen_grid
from lib.rpc_model import RPCModel

# latlon points, utm_points, local_points
def convert_to_points():
    zz = np.load(os.path.join(debug_dir, 'dsm_gt_data.npy'))    # (height, width)
    zz = zz.reshape((-1, 1))
    valid_mask = np.logical_not(np.isnan(zz))
    zz = zz[valid_mask].reshape((-1, 1))

    # latlon points
    with open(os.path.join(debug_dir, 'bbx_latlon.json')) as fp:
        bbx_latlon = json.load(fp)

    xx = np.linspace(bbx_latlon['ul_lat'], bbx_latlon['lr_lat'], bbx_latlon['img_height'])
    yy = np.linspace(bbx_latlon['ul_lon'], bbx_latlon['lr_lon'], bbx_latlon['img_width'])
    xx, yy = np.meshgrid(xx, yy, indexing='ij')     # xx, yy are of shape (height, width)
    xx = xx.reshape((-1, 1))
    yy = yy.reshape((-1, 1))

    xx = xx[valid_mask].reshape((-1, 1))
    yy = yy[valid_mask].reshape((-1, 1))

    latlon_points = np.concatenate((xx, yy, zz), axis=1)
    np.save(os.path.join(debug_dir, 'dsm_latlon_points.npy'), latlon_points)

    # utm points
    with open(os.path.join(debug_dir, 'bbx_utm.json')) as fp:
        bbx_utm = json.load(fp)

    # xx = np.linspace(bbx_utm['ul_northing'], bbx_utm['lr_northing'], bbx_utm['img_height'])
    # yy = np.linspace(bbx_utm['ul_easting'], bbx_utm['lr_easting'], bbx_utm['img_width'])
    # xx, yy = np.meshgrid(xx, yy, indexing='ij')     # xx, yy are of shape (height, width)
    # xx = xx.reshape((-1, 1))
    # yy = yy.reshape((-1, 1))

    # xx = xx[valid_mask].reshape((-1, 1))
    # yy = yy[valid_mask].reshape((-1, 1))


    yy, xx, zz = latlonalt_to_eastnorthalt(xx, yy, zz)

    utm_points = np.concatenate((yy, xx, zz), axis=1)
    np.save(os.path.join(debug_dir, 'dsm_utm_points.npy'), utm_points)

    # local utm points
    # ll_northing = bbx_utm['lr_northing']
    # ll_easting = bbx_utm['ul_easting']

    with open(os.path.join(debug_dir, 'aoi.json')) as fp:
        aoi_dict = json.load(fp)
    ll_easting = aoi_dict['ul_easting']
    ll_northing = aoi_dict['ul_northing'] - aoi_dict['height']

    xx = xx - ll_northing
    yy = yy - ll_easting
    utm_points_local = np.concatenate((yy, xx, zz), axis=1)
    np.save(os.path.join(debug_dir, 'dsm_utm_points_local.npy'), utm_points_local)

    # project to image space with RPC model
    with open(os.path.join(debug_dir, '000_20141115135121.json')) as fp:
        rpc = RPCModel(json.load(fp))
    col, row = rpc.projection(latlon_points[:, 0], latlon_points[:, 1], latlon_points[:, 2])

    pixels = np.concatenate((col[:, np.newaxis], row[:, np.newaxis]), axis=1)
    np.save(os.path.join(debug_dir, 'dsm_pixels.npy'), pixels)

#
# def discretize():
#     # each grid cell is about 5 meters * 5 meters * 5 meters
#     num_xy_points = 50
#     num_z_points = 20
#
#     with open(os.path.join(debug_dir, 'bbx_utm.json')) as fp:
#         bbx_utm = json.load(fp)
#
#     with open(os.path.join(debug_dir, 'bbx_latlon.json')) as fp:
#         bbx_latlon = json.load(fp)
#
#     lat_points = np.linspace(bbx_latlon['ul_lat'], bbx_latlon['lr_lat'], num_xy_points)
#     lon_points = np.linspace(bbx_latlon['ul_lon'], bbx_latlon['lr_lon'], num_xy_points)
#     z_points = np.linspace(bbx_latlon['z_min'], bbx_latlon['z_max'], num_z_points)
#
#     lat_points, lon_points, z_points = gen_grid(lat_points, lon_points, z_points)
#
#     # project to image space with RPC model
#     with open(os.path.join(debug_dir, '000_20141115135121.json')) as fp:
#         rpc = RPCModel(json.load(fp))
#     col, row = rpc.projection(lat_points, lon_points, z_points)
#     pixels = np.concatenate((col, row), axis=1)
#     np.save(os.path.join(debug_dir, 'grid_pixels.npy'), pixels)
#
#     valid_mask = col >= 0
#     valid_mask = np.logical_and(valid_mask, col < bbx_latlon['width'])
#     valid_mask = np.logical_and(valid_mask, row >= 0)
#     valid_mask = np.logical_and(valid_mask, row < bbx_latlon['height'])
#
#     np.save(os.path.join(debug_dir, 'grid_valid_mask.npy'), np.float32(valid_mask))
#
#     northing_points = np.linspace(bbx_utm['ul_northing'], bbx_utm['lr_northing'], num_xy_points)
#     easting_points = np.linspace(bbx_utm['ul_easting'], bbx_utm['lr_easting'], num_xy_points)
#     z_points = np.linspace(bbx_utm['z_min'], bbx_utm['z_max'], num_z_points)
#     northing_points, easting_points, z_points = gen_grid(northing_points, easting_points, z_points)
#
#     # local utm points
#     # ll_northing = bbx_utm['lr_northing']
#     # ll_easting = bbx_utm['ul_easting']
#
#     with open(os.path.join(debug_dir, 'aoi.json')) as fp:
#         aoi_dict = json.load(fp)
#     ll_easting = aoi_dict['ul_easting']
#     ll_northing = aoi_dict['lr_northing']
#
#     xx = easting_points - ll_easting
#     yy = northing_points - ll_northing
#     grid_points_local = np.concatenate((xx, yy, z_points), axis=1)
#     np.save(os.path.join(debug_dir, 'grid_points_local.npy'), grid_points_local)


def discretize():
    # each grid cell is about 5 meters * 5 meters * 5 meters
    num_xy_points = 100
    num_z_points = 20

    with open(os.path.join(debug_dir, 'bbx_utm.json')) as fp:
        bbx_utm = json.load(fp)
    img_width = bbx_utm['img_width']
    img_height = bbx_utm['img_height']

    # z_min = bbx_utm['z_min']
    # z_max = bbx_utm['z_max']

    z_min = 0
    z_max = 100

    with open(os.path.join(debug_dir, 'aoi.json')) as fp:
        aoi_dict = json.load(fp)
    ul_easting = aoi_dict['ul_easting']
    ul_northing = aoi_dict['ul_northing']
    lr_easting = ul_easting + aoi_dict['width']
    lr_northing = ul_northing - aoi_dict['height']
    ll_easting = ul_easting
    ll_northing = lr_northing
    zone_number = aoi_dict['zone_number']
    hemishpere = aoi_dict['hemisphere']
    northern = True if hemishpere == 'N' else False

    ul_lat, ul_lon = utm.to_latlon(ul_easting, ul_northing, zone_number, northern=northern)
    lr_lat, lr_lon = utm.to_latlon(lr_easting, lr_northing, zone_number, northern=northern)

    lat_points = np.linspace(ul_lat, lr_lat, num_xy_points)
    lon_points = np.linspace(ul_lon, lr_lon, num_xy_points)
    z_points = np.linspace(z_min, z_max, num_z_points)

    lat_points, lon_points, z_points = gen_grid(lat_points, lon_points, z_points)

    # project to image space with RPC model
    with open(os.path.join(debug_dir, '000_20141115135121.json')) as fp:
        rpc = RPCModel(json.load(fp))
    col, row = rpc.projection(lat_points, lon_points, z_points)
    pixels = np.concatenate((col, row), axis=1)
    np.save(os.path.join(debug_dir, 'grid_pixels.npy'), pixels)

    valid_mask = col >= 0
    valid_mask = np.logical_and(valid_mask, col < img_width)
    valid_mask = np.logical_and(valid_mask, row >= 0)
    valid_mask = np.logical_and(valid_mask, row < img_height)

    np.save(os.path.join(debug_dir, 'grid_valid_mask.npy'), np.float32(valid_mask))

    # northing_points = np.linspace(ul_northing, lr_northing, num_xy_points)
    # easting_points = np.linspace(ul_easting, lr_easting, num_xy_points)
    # z_points = np.linspace(z_min, z_max, num_z_points)
    # northing_points, easting_points, z_points = gen_grid(northing_points, easting_points, z_points)

    # cnt = lat_points.shape[0]
    # easting_points = np.zeros((cnt, 1))
    # northing_points = np.zeros((cnt, 1))
    # for i in range(cnt):
    #     tmp = utm.from_latlon(lat_points[i, 0], lon_points[i, 0])
    #     easting_points[i, 0] = tmp[0]
    #     northing_points[i, 0] = tmp[1]

    easting_points, northing_points, z_points = latlonalt_to_eastnorthalt(lat_points, lon_points, z_points)

    # local utm points
    xx = easting_points - ll_easting
    yy = northing_points - ll_northing
    grid_points_local = np.concatenate((xx, yy, z_points), axis=1)
    np.save(os.path.join(debug_dir, 'grid_points_local.npy'), grid_points_local)


def random_sample():
    # each grid cell is about 5 meters * 5 meters * 5 meters
    num_xy_points = 50
    num_z_points = 20

    with open(os.path.join(debug_dir, 'bbx_utm.json')) as fp:
        bbx_utm = json.load(fp)
    img_width = bbx_utm['img_width']
    img_height = bbx_utm['img_height']

    # z_min = bbx_utm['z_min']
    # z_max = bbx_utm['z_max']

    z_min = 0
    z_max = 100

    with open(os.path.join(debug_dir, 'aoi.json')) as fp:
        aoi_dict = json.load(fp)
    ul_easting = aoi_dict['ul_easting']
    ul_northing = aoi_dict['ul_northing']
    lr_easting = ul_easting + aoi_dict['width']
    lr_northing = ul_northing - aoi_dict['height']
    ll_easting = ul_easting
    ll_northing = lr_northing
    zone_number = aoi_dict['zone_number']
    hemishpere = aoi_dict['hemisphere']
    northern = True if hemishpere == 'N' else False

    ul_lat, ul_lon = utm.to_latlon(ul_easting, ul_northing, zone_number, northern=northern)
    lr_lat, lr_lon = utm.to_latlon(lr_easting, lr_northing, zone_number, northern=northern)

    lat_points = ul_lat + np.random.uniform(size=num_xy_points) * (lr_lat - ul_lat)
    lon_points = ul_lon + np.random.uniform(size=num_xy_points) * (lr_lon - ul_lon)
    z_points = z_min + np.random.uniform(size=num_z_points) * (z_max - z_min)

    lat_points, lon_points, z_points = gen_grid(lat_points, lon_points, z_points)

    # project to image space with RPC model
    with open(os.path.join(debug_dir, '000_20141115135121.json')) as fp:
        rpc = RPCModel(json.load(fp))
    col, row = rpc.projection(lat_points, lon_points, z_points)

    # remove invalid pixels
    valid_mask = col >= 0
    valid_mask = np.logical_and(valid_mask, col < img_width)
    valid_mask = np.logical_and(valid_mask, row >= 0)
    valid_mask = np.logical_and(valid_mask, row < img_height)

    lat_points = lat_points[valid_mask].reshape((-1, 1))
    lon_points = lon_points[valid_mask].reshape((-1, 1))
    z_points = z_points[valid_mask].reshape((-1, 1))
    col = col[valid_mask].reshape((-1, 1))
    row = row[valid_mask].reshape((-1, 1))

    pixels = np.concatenate((col, row), axis=1)
    np.save(os.path.join(debug_dir, 'random_pixels.npy'), pixels)

    # cnt = lat_points.shape[0]
    # easting_points = np.zeros((cnt, 1))
    # northing_points = np.zeros((cnt, 1))
    # for i in range(cnt):
    #     tmp = utm.from_latlon(lat_points[i, 0], lon_points[i, 0])
    #     easting_points[i, 0] = tmp[0]
    #     northing_points[i, 0] = tmp[1]

    easting_points, northing_points, z_points = latlonalt_to_eastnorthalt(lat_points, lon_points, z_points)

    # local utm points
    xx = easting_points - ll_easting
    yy = northing_points - ll_northing
    random_points_local = np.concatenate((xx, yy, z_points), axis=1)
    np.save(os.path.join(debug_dir, 'random_points_local.npy'), random_points_local)


from lib.solve_perspective import solve_perspective
def derive_approx():
    grid_points_local = np.load(os.path.join(debug_dir, 'grid_points_local.npy'))
    grid_pixels = np.load(os.path.join(debug_dir, 'grid_pixels.npy'))
    grid_valid_mask = np.load(os.path.join(debug_dir, 'grid_valid_mask.npy')) > 0

    K, R, t = solve_perspective(grid_points_local[:, 0:1], grid_points_local[:, 1:2], grid_points_local[:, 2:3],
                                grid_pixels[:, 0:1], grid_pixels[:, 1:2], grid_valid_mask)

    P = np.dot(K, np.hstack((R, t)))
    np.savetxt(os.path.join(debug_dir, 'approx_proj_mat.txt'), P, delimiter=',')

    cnt = grid_points_local.shape[0]
    grid_points_local = np.concatenate((grid_points_local, np.ones((cnt, 1))), axis=1)

    tmp = np.dot(grid_points_local, P.T)
    grid_pixels_approx = np.concatenate((tmp[:, 0:1] / tmp[:, 2:3], tmp[:, 1:2] / tmp[:, 2:3]), axis=1)

    # compute err
    err = np.sqrt(np.sum((grid_pixels - grid_pixels_approx) ** 2, axis=1))
    print('{} grid points, err, min: {}, max: {}, median: {}, mean: {}'.format(cnt, np.min(err),
                                                                               np.max(err),
                                                                               np.median(err), np.mean(err)))

    random_points_local = np.load(os.path.join(debug_dir, 'random_points_local.npy'))
    cnt = random_points_local.shape[0]
    random_points_local = np.concatenate((random_points_local, np.ones((cnt, 1))), axis=1)

    random_pixels = np.load(os.path.join(debug_dir, 'random_pixels.npy'))
    tmp = np.dot(random_points_local, P.T)
    random_pixels_approx = np.concatenate((tmp[:, 0:1] / tmp[:, 2:3], tmp[:, 1:2] / tmp[:, 2:3]), axis=1)

    err = np.sqrt(np.sum((random_pixels - random_pixels_approx) ** 2, axis=1))
    print('{} random points, err, min: {}, max: {}, median: {}, mean: {}'.format(cnt, np.min(err),
                                                                               np.max(err),
                                                                               np.median(err), np.mean(err)))

# col, row
def check_error():
    P = np.loadtxt(os.path.join(debug_dir, 'approx_proj_mat.txt'), delimiter=',')

    utm_points_local = np.load(os.path.join(debug_dir, 'dsm_utm_points_local.npy'))
    pixels = np.load(os.path.join(debug_dir, 'dsm_pixels.npy'))

    cnt = utm_points_local.shape[0]
    utm_points_local = np.concatenate((utm_points_local, np.ones((cnt, 1))), axis=1)

    tmp = np.dot(utm_points_local, P.T)
    pixels_approx = np.concatenate((tmp[:, 0:1] / tmp[:, 2:3], tmp[:, 1:2] / tmp[:, 2:3]), axis=1)

    # compute err
    err = np.sqrt(np.sum((pixels - pixels_approx) ** 2, axis=1))
    print('{} dsm points, err, min: {}, max: {}, median: {}, mean: {}'.format(cnt, np.min(err),
                                                                               np.max(err),
                                                                               np.median(err), np.mean(err)))


if __name__ == '__main__':
    process_dsm_gt()
    gen_bbx()
    convert_to_points()
    discretize()
    random_sample()
    derive_approx()
    check_error()
