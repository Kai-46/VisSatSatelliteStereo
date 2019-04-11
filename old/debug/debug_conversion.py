import numpy as np
import re
from old.image_util import read_image
import utm
from lib.save_image_only import save_image_only
from lib.latlon_utm_converter import latlon_to_eastnorh


def create_grid(gt_file):
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

    # derive bounding box
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

    # latlon points
    xx = np.linspace(bbx_latlon['ul_lat'], bbx_latlon['lr_lat'], bbx_latlon['img_height'])
    yy = np.linspace(bbx_latlon['ul_lon'], bbx_latlon['lr_lon'], bbx_latlon['img_width'])
    lat, lon = np.meshgrid(xx, yy, indexing='ij')     # xx, yy are of shape (height, width)

    xx = lat.reshape((-1, 1))
    yy = lon.reshape((-1, 1))

    east, north = latlon_to_eastnorh(xx, yy)
    size = (bbx_latlon['img_height'], bbx_latlon['img_width'])
    east = east.reshape(size)
    north = north.reshape(size)

    # utm points
    xx = np.linspace(bbx_utm['ul_northing'], bbx_utm['lr_northing'], bbx_utm['img_height'])
    yy = np.linspace(bbx_utm['ul_easting'], bbx_utm['lr_easting'], bbx_utm['img_width'])
    north_new, east_new = np.meshgrid(xx, yy, indexing='ij')     # xx, yy are of shape (height, width)

    save_image_only(north - north_new, '/data2/debug_north.jpg', save_cbar=True)
    save_image_only(east - east_new, '/data2/debug_east.jpg', save_cbar=True)

    # check upper-right corner
    ur_lat = ul_lat
    ur_lon = lr_lon

    ur_easting = lr_easting
    ur_northing = ul_northing

    tmp = utm.from_latlon(ur_lat, ur_lon)
    ur_easting_new = tmp[0]
    ur_northing_new = tmp[1]
    print('convert from lat, lon: ({}, {}), original: ({}, {})'.format(ur_easting_new, ur_northing_new,
                                                                       ur_easting, ur_northing))

    # check lower-left corner
    ll_lat = lr_lat
    ll_lon = ul_lon

    ll_easting = ul_easting
    ll_northing = lr_northing

    tmp = utm.from_latlon(ll_lat, ll_lon)
    ll_easting_new = tmp[0]
    ll_northing_new = tmp[1]
    print('convert from lat, lon: ({}, {}), original: ({}, {})'.format(ll_easting_new, ll_northing_new,
                                                                       ll_easting, ll_northing))



if __name__ == '__main__':
    gt_file = '/bigdata/kz298/dataset/mvs3dm/Challenge_Data_and_Software/Lidar_gt/MasterSequesteredPark.tif'
    create_grid(gt_file)
