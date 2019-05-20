import os
import numpy as np
import shutil
from colmap.read_dense import read_array
from lib.proj_to_utm_grid import proj_to_utm_grid
from lib.proj_to_grid import proj_to_grid
from visualization.plot_height_map import plot_height_map
from visualization.save_image_only import save_image_only
import json
import logging
import multiprocessing
from coordinate_system import local_to_global


def convert_depth_map_worker(work_dir, out_dir, items, depth_type):
    mvs_dir = os.path.join(work_dir, 'colmap/mvs')

    # load inv_proj_mats
    inv_proj_mats = {}
    with open(os.path.join(mvs_dir, 'inv_proj_mats.txt')) as fp:
        for line in fp.readlines():
            tmp = line.split(' ')
            img_name = tmp[0]
            mats = np.array([float(tmp[i]) for i in range(1, 17)]).reshape((4, 4))
            inv_proj_mats[img_name] = mats

    with open(os.path.join(work_dir, 'ground_truth/dsm_gt_meta.json')) as fp:
        meta_dict = json.load(fp)

    # then read the depth maps
    depth_dir = os.path.join(mvs_dir, 'stereo/depth_maps')
    image_dir = os.path.join(mvs_dir, 'images')
    for item in items:
        idx = item.rfind('.{}.bin'.format(depth_type))
        if idx == -1:
            continue

        logging.info('converting depth: {}'.format(item))

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
        tmp = np.vstack((col, row, np.ones((1, width * height)), depth))
        tmp = np.dot(inv_proj_mats[img_name], tmp)
        tmp[0, :] /= tmp[3, :]
        tmp[1, :] /= tmp[3, :]
        tmp[2, :] /= tmp[3, :]
        height_map = tmp[2:3, :].reshape((height, width))

        # copy raw image and save height map
        shutil.copy2(os.path.join(image_dir, img_name), os.path.join(out_dir, 'img_grid'))
        np.save(os.path.join(out_dir, 'img_grid_npy/{}.{}.height.npy'.format(img_name, depth_type)),
                height_map)

        plot_height_map(height_map,
                        os.path.join(out_dir, 'img_grid/{}.{}.height.jpg'.format(img_name, depth_type)))

        # geo grid
        valid_mask = depth > 0.0
        xx = tmp[0:1, :][valid_mask].reshape((-1, 1))
        yy = tmp[1:2, :][valid_mask].reshape((-1, 1))
        zz = tmp[2:3, :][valid_mask].reshape((-1, 1))

        # np.save(os.path.join(out_dir, 'enu_points/{}.{}.points.npy'.format(img_name, depth_type)),
        #         np.hstack((xx, yy, zz)))

        with open(os.path.join(work_dir, 'geo_grid.json')) as fp:
            geo_grid = json.load(fp)
        dsm = proj_to_grid(np.hstack((xx, yy, zz)), geo_grid['ul_e'], geo_grid['ul_n'],
                           geo_grid['e_resolution'], geo_grid['n_resolution'], geo_grid['e_size'], geo_grid['n_size'])
        np.save(os.path.join(out_dir, 'enu_grid_npy/{}.{}.dsm.npy'.format(img_name, depth_type)),
                dsm)
        plot_height_map(dsm,
                        os.path.join(out_dir, 'enu_grid/{}.{}.dsm.jpg'.format(img_name, depth_type)),
                        save_cbar=True)

        lat, lon, alt = local_to_global(work_dir, xx, yy, zz)
        dsm = proj_to_utm_grid(np.hstack((lat, lon, alt)), meta_dict['ul_easting'], meta_dict['ul_northing'],
                                   meta_dict['east_resolution'], meta_dict['north_resolution'],
                                   meta_dict['img_width'], meta_dict['img_height'])
        np.save(os.path.join(out_dir, 'geo_grid_npy/{}.{}.dsm.npy'.format(img_name, depth_type)),
                dsm)
        plot_height_map(dsm,
                        os.path.join(out_dir, 'geo_grid/{}.{}.dsm.jpg'.format(img_name, depth_type)),
                        save_cbar=True, force_range=(meta_dict['alt_min'], meta_dict['alt_max']))


def convert_depth_maps(work_dir, depth_type):
    mvs_dir = os.path.join(work_dir, 'colmap/mvs')

    mvs_result_dir = os.path.join(work_dir, 'mvs_results')
    if not os.path.exists(mvs_result_dir):
        os.mkdir(mvs_result_dir)

    out_dir = os.path.join(mvs_result_dir, 'height_maps')
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    for subdir in [out_dir,
                   os.path.join(out_dir, 'img_grid'),
                   os.path.join(out_dir, 'geo_grid'),
                   os.path.join(out_dir, 'img_grid_npy'),
                   os.path.join(out_dir, 'geo_grid_npy'),
                   os.path.join(out_dir, 'enu_grid'),
                   os.path.join(out_dir, 'enu_grid_npy')]:
        if not os.path.exists(subdir):
            os.mkdir(subdir)

    # copy ground truth dsm into geo_grid folder
    shutil.copy2(os.path.join(work_dir, 'ground_truth/dsm_gt.jpg'), os.path.join(out_dir, 'geo_grid'))

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
        p = multiprocessing.Process(target=convert_depth_map_worker,
                                    args=(work_dir, out_dir, all_items[idx1:idx2], depth_type))
        process_list.append(p)
        p.start()

    for p in process_list:
        p.join()


def convert_normal_maps(work_dir, normal_type):
    mvs_dir = os.path.join(work_dir, 'colmap/mvs')
    out_dir = os.path.join(work_dir, 'mvs_results/normal_maps')
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
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
    pass

