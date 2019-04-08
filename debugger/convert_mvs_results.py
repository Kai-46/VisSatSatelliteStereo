import os
import numpy as np
import shutil
from colmap.read_dense import read_array
from lib.proj_to_geo_grid import proj_to_geo_grid
from lib.save_image_only import save_image_only
import json
import logging
import multiprocessing


def depth_map_converter(tile_dir, out_dir, items, depth_type):
    mvs_dir = os.path.join(tile_dir, 'colmap/mvs')

    # load inv_proj_mats
    inv_proj_mats = {}
    with open(os.path.join(mvs_dir, 'inv_proj_mats.txt')) as fp:
        for line in fp.readlines():
            tmp = line.split(' ')
            img_name = tmp[0]
            mats = np.array([float(tmp[i]) for i in range(1, 17)]).reshape((4, 4))
            inv_proj_mats[img_name] = mats

    with open(os.path.join(tile_dir, 'ground_truth/dsm_gt_bbx_local.json')) as fp:
        bbx_local = json.load(fp)

    # then read the depth maps
    depth_dir = os.path.join(mvs_dir, 'stereo/depth_maps')
    image_dir = os.path.join(mvs_dir, 'images')
    for item in items:
        idx = item.rfind('.{}.bin'.format(depth_type))
        if idx == -1:
            continue

        img_name = item[:idx]

        depth_map = read_array(os.path.join(depth_dir, item))

        depth_map[depth_map <= 0.0] = np.nan    # actually it is -1e20

        # create a meshgrid
        height, width = depth_map.shape
        row, col = np.meshgrid(range(height), range(width), indexing='ij')
        col = col.reshape((1, -1))
        row = row.reshape((1, -1))
        depth = depth_map.reshape((1, -1))

        # image grid
        tmp = np.vstack((col + 0.5, row + 0.5, np.ones((1, width * height)), depth))
        tmp = np.dot(inv_proj_mats[img_name], tmp)
        tmp[0, :] /= tmp[3, :]
        tmp[1, :] /= tmp[3, :]
        tmp[2, :] /= tmp[3, :]
        height_map = tmp[2:3, :].reshape((height, width))

        # copy raw image and save height map
        shutil.copy2(os.path.join(image_dir, img_name), os.path.join(out_dir, 'img_grid'))
        np.save(os.path.join(out_dir, 'img_grid_npy/{}.{}.height.npy'.format(img_name, depth_type)),
                height_map)
        nan_mask = np.isnan(height_map)
        height_map[nan_mask] = np.nanmin(height_map)
        save_image_only(1.0-np.float32(nan_mask),
                        os.path.join(out_dir, 'img_grid/{}.{}.height_mask.jpg'.format(img_name, depth_type)), plot=False)
        logging.info('height map, min: {}, max: {}'.format(height_map.min(), height_map.max()))
        save_image_only(height_map,
                        os.path.join(out_dir, 'img_grid/{}.{}.height.jpg'.format(img_name, depth_type)))

        # geo grid
        valid_mask = depth > 0.0
        x = tmp[0:1, :][valid_mask].reshape((1, -1))
        y = tmp[1:2, :][valid_mask].reshape((1, -1))
        z = tmp[2:3, :][valid_mask].reshape((1, -1))
        points = np.vstack((x, y, z)).T
        dsm = proj_to_geo_grid(points, bbx_local['ul_easting'], bbx_local['ul_northing'],
                               bbx_local['resolution'], bbx_local['img_width'], bbx_local['img_height'])
        np.save(os.path.join(out_dir, 'geo_grid_npy/{}.{}.dsm.npy'.format(img_name, depth_type)),
                dsm)
        nan_mask = np.isnan(dsm)
        dsm[nan_mask] = np.nanmin(dsm)
        save_image_only(1.0-np.float32(nan_mask),
                        os.path.join(out_dir, 'geo_grid/{}.{}.dsm_mask.jpg'.format(img_name, depth_type)), plot=False)
        save_image_only(dsm,
                        os.path.join(out_dir, 'geo_grid/{}.{}.dsm.jpg'.format(img_name, depth_type)))


def convert_depth_maps(tile_dir, depth_type):
    mvs_dir = os.path.join(tile_dir, 'colmap/mvs')

    mvs_result_dir = os.path.join(tile_dir, 'mvs_results')
    if not os.path.exists(mvs_result_dir):
        os.mkdir(mvs_result_dir)
    out_dir = os.path.join(mvs_result_dir, 'height_maps')
    for subdir in [out_dir,
                   os.path.join(out_dir, 'img_grid'),
                   os.path.join(out_dir, 'geo_grid'),
                   os.path.join(out_dir, 'img_grid_npy'),
                   os.path.join(out_dir, 'geo_grid_npy')]:
        if not os.path.exists(subdir):
            os.mkdir(subdir)

    # copy ground truth dsm into geo_grid folder
    shutil.copy2(os.path.join(tile_dir, 'ground_truth/dsm_gt.jpg'), os.path.join(out_dir, 'geo_grid'))

    # all items to be converted
    depth_dir = os.path.join(mvs_dir, 'stereo/depth_maps')
    all_items = []
    for item in sorted(os.listdir(depth_dir)):
        idx = item.rfind('.{}.bin'.format(depth_type))
        if idx >= 0:
            all_items.append(item)

    # split all_tracks into multiple chunks
    process_cnt = multiprocessing.cpu_count()
    process_list = []

    chunk_size = int(len(all_items) / process_cnt)
    chunks = [[j*chunk_size, (j+1)*chunk_size] for j in range(process_cnt)]
    chunks[-1][1] = len(all_items)
    for i in range(process_cnt):
        idx1 = chunks[i][0]
        idx2 = chunks[i][1]
        p = multiprocessing.Process(target=depth_map_converter,
                                    args=(tile_dir, out_dir, all_items[idx1:idx2], depth_type))
        process_list.append(p)
        p.start()

    for p in process_list:
        p.join()


# def convert_depth_maps(tile_dir, depth_type):
#     mvs_dir = os.path.join(tile_dir, 'colmap/mvs')
#
#     mvs_result_dir = os.path.join(tile_dir, 'mvs_results')
#     if not os.path.exists(mvs_result_dir):
#         os.mkdir(mvs_result_dir)
#     out_dir = os.path.join(mvs_result_dir, 'height_maps')
#     for subdir in [out_dir,
#                    os.path.join(out_dir, 'img_grid'),
#                    os.path.join(out_dir, 'geo_grid'),
#                    os.path.join(out_dir, 'img_grid_npy'),
#                    os.path.join(out_dir, 'geo_grid_npy')]:
#         if not os.path.exists(subdir):
#             os.mkdir(subdir)
#
#     # load inv_proj_mats
#     inv_proj_mats = {}
#     with open(os.path.join(mvs_dir, 'inv_proj_mats.txt')) as fp:
#         for line in fp.readlines():
#             tmp = line.split(' ')
#             img_name = tmp[0]
#             mats = np.array([float(tmp[i]) for i in range(1, 17)]).reshape((4, 4))
#             inv_proj_mats[img_name] = mats
#
#     # copy ground truth dsm into geo_grid folder
#     shutil.copy2(os.path.join(tile_dir, 'ground_truth/dsm_gt.jpg'), os.path.join(out_dir, 'geo_grid'))
#
#     with open(os.path.join(tile_dir, 'ground_truth/dsm_gt_bbx_local.json')) as fp:
#         bbx_local = json.load(fp)
#
#     # then read the depth maps
#     depth_dir = os.path.join(mvs_dir, 'stereo/depth_maps')
#     image_dir = os.path.join(mvs_dir, 'images')
#     for item in sorted(os.listdir(depth_dir)):
#         idx = item.rfind('.{}.bin'.format(depth_type))
#         if idx == -1:
#             continue
#
#         img_name = item[:idx]
#
#         depth_map = read_array(os.path.join(depth_dir, item))
#
#         depth_map[depth_map <= 0.0] = np.nan    # actually it is -1e20
#
#         # create a meshgrid
#         height, width = depth_map.shape
#         row, col = np.meshgrid(range(height), range(width), indexing='ij')
#         col = col.reshape((1, -1))
#         row = row.reshape((1, -1))
#         depth = depth_map.reshape((1, -1))
#
#         # image grid
#         tmp = np.vstack((col + 0.5, row + 0.5, np.ones((1, width * height)), depth))
#         tmp = np.dot(inv_proj_mats[img_name], tmp)
#         tmp[0, :] /= tmp[3, :]
#         tmp[1, :] /= tmp[3, :]
#         tmp[2, :] /= tmp[3, :]
#         height_map = tmp[2:3, :].reshape((height, width))
#
#         # copy raw image and save height map
#         shutil.copy2(os.path.join(image_dir, img_name), os.path.join(out_dir, 'img_grid'))
#         np.save(os.path.join(out_dir, 'img_grid_npy/{}.{}.height.npy'.format(img_name, depth_type)),
#                 height_map)
#         nan_mask = np.isnan(height_map)
#         height_map[nan_mask] = np.nanmin(height_map)
#         save_image_only(1.0-np.float32(nan_mask),
#                         os.path.join(out_dir, 'img_grid/{}.{}.height_mask.jpg'.format(img_name, depth_type)), plot=False)
#         logging.info('height map, min: {}, max: {}'.format(height_map.min(), height_map.max()))
#         save_image_only(height_map,
#                         os.path.join(out_dir, 'img_grid/{}.{}.height.jpg'.format(img_name, depth_type)))
#
#         # geo grid
#         valid_mask = depth > 0.0
#         x = tmp[0:1, :][valid_mask].reshape((1, -1))
#         y = tmp[1:2, :][valid_mask].reshape((1, -1))
#         z = tmp[2:3, :][valid_mask].reshape((1, -1))
#         points = np.vstack((x, y, z)).T
#         dsm = proj_to_geo_grid(points, bbx_local['ul_easting'], bbx_local['ul_northing'],
#                                bbx_local['resolution'], bbx_local['img_width'], bbx_local['img_height'])
#         np.save(os.path.join(out_dir, 'geo_grid_npy/{}.{}.dsm.npy'.format(img_name, depth_type)),
#                 dsm)
#         nan_mask = np.isnan(dsm)
#         dsm[nan_mask] = np.nanmin(dsm)
#         save_image_only(1.0-np.float32(nan_mask),
#                         os.path.join(out_dir, 'geo_grid/{}.{}.dsm_mask.jpg'.format(img_name, depth_type)), plot=False)
#         save_image_only(dsm,
#                         os.path.join(out_dir, 'geo_grid/{}.{}.dsm.jpg'.format(img_name, depth_type)))


def convert_normal_maps(tile_dir, normal_type):
    mvs_dir = os.path.join(tile_dir, 'colmap/mvs')
    out_dir = os.path.join(tile_dir, 'mvs_results/normal_maps')
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    normal_dir = os.path.join(mvs_dir, 'stereo/normal_maps')
    for item in sorted(os.listdir(normal_dir)):
        idx = item.rfind('.{}.bin'.format(normal_type))
        if idx == -1:
            continue

        img_name = item[:idx]
        normal_file = os.path.join(normal_dir, '{}.{}.bin'.format(img_name, normal_type))

        normal_map = read_array(normal_file)

        # filter absurd value
        normal_map[normal_map < -1e19] = -1.0

        normal_map = (normal_map + 1.0) / 2.0

        save_image_only(normal_map,
                        os.path.join(out_dir, '{}.{}.normal.jpg'.format(img_name, normal_type)), plot=False)


if __name__ == '__main__':
    mvs_dir = '/data2/kz298/mvs3dm_result/MasterSequesteredPark/colmap/mvs'
    #mvs_dir = '/data2/kz298/core3d_result/aoi-d4-jacksonville/colmap/mvs'

    #mvs_dir = '/home/kai/satellite_stereo/Explorer_subset/colmap/mvs'
    out_dir = os.path.join(mvs_dir, 'height_maps')

    #convert_depth_maps(mvs_dir, out_dir, 'photometric')
    convert_depth_maps(mvs_dir, out_dir, 'geometric')

    #convert_normal_maps(mvs_dir, out_dir, 'photometric')
    convert_normal_maps(mvs_dir, out_dir, 'geometric')
