import os
import json
import numpy as np
import re
from lib.image_util import read_image
from lib.save_image_only import save_image_only
from lib.latlon_utm_converter import eastnorth_to_latlon


def center_crop(nan_mask):
    height, width = nan_mask.shape
    thres = 0.7

    for col_min in range(width):
        empty_ratio = np.sum(nan_mask[:, col_min]) / height
        if empty_ratio < thres:
            break

    for col_max in range(width-1, -1, -1):
        if np.sum(nan_mask[:, col_max]) / height < thres:
            break

    for row_min in range(height):
        if np.sum(nan_mask[row_min, :]) / width < thres:
            break

    for row_max in range(height-1, -1, -1):
        if np.sum(nan_mask[row_max, :]) / width < thres:
            break

    return (col_min, col_max, row_min, row_max)


def process_dsm_gt(tile_dir, gt_file):
    gt_subdir = os.path.join(tile_dir, 'ground_truth')
    if not os.path.exists(gt_subdir):
        os.mkdir(gt_subdir)

    gt_dsm, geo, proj, meta, width, height = read_image(gt_file)
    gt_dsm = gt_dsm[:, :, 0]
    nan_mask = gt_dsm < -9998   # actual nan value is -9999
    gt_dsm[nan_mask] = np.nan
    # center-crop
    # col_min, col_max, row_min, row_max = center_crop(nan_mask)
    # col_min -= 30
    # col_max += 30
    # row_min -= 30
    # row_max += 30
    # print('{}, {}, {}, {}'.format(col_min, col_max, row_min, row_max))

    # gt_dsm = gt_dsm[row_min:row_max+1, col_min:col_max+1]
    # nan_mask = nan_mask[row_min:row_max+1, col_min:col_max+1]

    utm_zone_str = re.findall('WGS 84 / UTM zone [0-9]{1,2}[N,S]{1}', proj)[0]
    hemisphere = utm_zone_str[-1]
    idx = utm_zone_str.rfind(' ')
    zone_number = int(utm_zone_str[idx:-1])

    # tif_meta_dict = {'ul_easting': geo[0]+geo[1]*col_min,
    #                  'ul_northing': geo[3]-geo[1]*row_min,
    #                  'resolution': geo[1],
    #                  'zone_number': zone_number,
    #                  'hemisphere': hemisphere,
    #                  'width': col_max - col_min + 1,
    #                  'height': row_max - row_min + 1,
    #                  'dsm_min': np.nanmin(gt_dsm).item(),
    #                  'dsm_max': np.nanmax(gt_dsm).item()}

    tif_meta_dict = {'ul_easting': geo[0],
                     'ul_northing': geo[3],
                     'resolution': geo[1],
                     'zone_number': zone_number,
                     'hemisphere': hemisphere,
                     'width': width,
                     'height': height,
                     'dsm_min': np.nanmin(gt_dsm).item(),
                     'dsm_max': np.nanmax(gt_dsm).item()}

    with open(os.path.join(gt_subdir, 'dsm_gt_meta.json'), 'w') as fp:
        json.dump(tif_meta_dict, fp, indent=2)

    # save data to npy
    image = np.copy(gt_dsm)
    image[nan_mask] = np.nanmin(image)
    save_image_only(image, os.path.join(gt_subdir, 'dsm_gt.jpg'))
    save_image_only(1.0-np.float32(nan_mask), os.path.join(gt_subdir, 'dsm_gt_valid_mask.jpg'), plot=False)
    np.save(os.path.join(gt_subdir, 'dsm_gt_data.npy'), gt_dsm)
    np.save(os.path.join(gt_subdir, 'dsm_gt_data_valid_mask.npy'), 1.0-np.float32(nan_mask))

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
        'img_height': height,
        'resolution': tif_meta_dict['resolution']
    }
    with open(os.path.join(gt_subdir, 'dsm_gt_bbx_utm.json'), 'w') as fp:
        json.dump(bbx_utm, fp, indent=2)

    # generate points
    zz = np.load(os.path.join(gt_subdir, 'dsm_gt_data.npy'))    # (height, width)
    zz = zz.reshape((-1, 1))
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
    np.save(os.path.join(gt_subdir, 'dsm_gt_utm_points.npy'), utm_points)

    # latlon points
    lat, lon = eastnorth_to_latlon(yy, xx, bbx_utm['zone_number'], bbx_utm['hemisphere'])
    latlon_points = np.concatenate((lat, lon, zz), axis=1)
    np.save(os.path.join(gt_subdir, 'dsm_gt_latlon_points.npy'), latlon_points)

    # bbx latlon
    lat_min = np.min(lat)
    lat_max = np.max(lat)
    lon_min = np.min(lon)
    lon_max = np.max(lon)

    bbx_latlon = {
        'lat_min': lat_min,
        'lat_max': lat_max,
        'lon_min': lon_min,
        'lon_max': lon_max,
        'lat_range': lat_max - lat_min,
        'lon_range': lon_max - lon_min,
        'z_min': z_min,
        'z_max': z_max,
        'img_width': width,
        'img_height': height,
        'resolution': tif_meta_dict['resolution']
    }
    with open(os.path.join(gt_subdir, 'dsm_gt_bbx_latlon.json'), 'w') as fp:
        json.dump(bbx_latlon, fp, indent=2)

    # local coordinate frame
    with open(os.path.join(tile_dir, 'aoi.json')) as fp:
        aoi_dict = json.load(fp)
    aoi_ll_easting = aoi_dict['ul_easting']
    aoi_ll_northing = aoi_dict['ul_northing'] - aoi_dict['height']

    bbx_local = {
        'ul_easting': bbx_utm['ul_easting'] - aoi_ll_easting,
        'ul_northing': bbx_utm['ul_northing'] - aoi_ll_northing,
        'lr_easting': bbx_utm['lr_easting'] - aoi_ll_easting,
        'lr_northing': bbx_utm['lr_northing'] - aoi_ll_northing,
        'z_min': bbx_utm['z_min'],
        'z_max': bbx_utm['z_max'],
        'zone_number': bbx_utm['zone_number'],
        'hemisphere': bbx_utm['hemisphere'],
        'img_width': bbx_utm['img_width'],
        'img_height': bbx_utm['img_height'],
        'resolution': tif_meta_dict['resolution']
    }
    with open(os.path.join(gt_subdir, 'dsm_gt_bbx_local.json'), 'w') as fp:
        json.dump(bbx_local, fp, indent=2)


if __name__ == '__main__':
    tile_dir = '/data2/kz298/mvs3dm_result/MasterProvisional2'
    # gt_file = '/bigdata/kz298/dataset/mvs3dm/Challenge_Data_and_Software/Lidar_gt/MasterProvisional2.tif'
    # process_dsm_gt(tile_dir, gt_file)

    col_min, col_max, row_min, row_max = (825, 2146, 480, 1739)
    import imageio
    im = imageio.imread(os.path.join(tile_dir, 'mvs_results/merged_dsm.jpg'))
    im = im[row_min:row_max+1, col_min:col_max+1, :]
    imageio.imwrite(os.path.join(tile_dir, 'mvs_results/merged_dsm_crop.jpg'), im)

    im = imageio.imread(os.path.join(tile_dir, 'mvs_results/colmap_fused_dsm.jpg'))
    im = im[row_min:row_max+1, col_min:col_max+1, :]
    imageio.imwrite(os.path.join(tile_dir, 'mvs_results/colmap_fused_dsm_crop.jpg'), im)
