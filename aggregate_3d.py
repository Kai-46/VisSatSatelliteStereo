from lib.run_cmd import run_cmd
import os
from lib.ply_np_converter import ply2np, np2ply
import json
from lib.proj_to_utm_grid import proj_to_utm_grid
from coordinate_system import local_to_global
import numpy as np
from visualization.plot_height_map import plot_height_map
from lib.dsm_util import write_dsm_tif
from lib.latlon_utm_converter import latlon_to_eastnorh
from lib.proj_to_grid import proj_to_grid


def fuse(colmap_dir):
    cmd = 'colmap stereo_fusion --workspace_path {colmap_dir}/mvs \
                         --output_path {colmap_dir}/mvs/fused.ply \
                         --input_type geometric \
                         --StereoFusion.min_num_pixels 3\
                         --StereoFusion.max_reproj_error 1\
                         --StereoFusion.max_depth_error 0.4\
                         --StereoFusion.max_normal_error 10'.format(colmap_dir=colmap_dir)
    run_cmd(cmd)


def run_fuse(work_dir):
    fuse(os.path.join(work_dir, 'colmap'))

    out_dir = os.path.join(work_dir, 'mvs_results/aggregate_3d')
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    points = ply2np(os.path.join(work_dir, 'colmap/mvs/fused.ply'))

    # proj to enu grid
    with open(os.path.join(work_dir, 'ground_truth/dsm_gt_bbx_enu.json')) as fp:
        bbx_enu = json.load(fp)
    e_min = bbx_enu['e_min']
    e_max = bbx_enu['e_max']
    n_min = bbx_enu['n_min']
    n_max = bbx_enu['n_max']
    resolution = 0.3
    e_size = int((e_max - e_min) / resolution) + 1
    n_size = int((n_max - n_min) / resolution) + 1
    dsm = proj_to_grid(points, e_min, n_max, resolution, resolution, e_size, n_size, propagate=True)

    plot_height_map(dsm, os.path.join(out_dir, 'enu_dsm.jpg'), save_cbar=True)

    void_ratio = np.sum(np.isnan(dsm)) / dsm.size
    with open(os.path.join(out_dir, 'enu_dsm_info.txt'), 'w') as fp:
        fp.write('completeness: {} %'.format((1.0 - void_ratio) * 100.0))

    lat, lon, alt = local_to_global(work_dir, points[:, 0:1], points[:, 1:2], points[:, 2:3])
    with open(os.path.join(work_dir, 'ground_truth/dsm_gt_meta.json')) as fp:
        meta_dict = json.load(fp)

    dsm = proj_to_utm_grid(np.hstack((lat, lon, alt)), meta_dict['ul_easting'], meta_dict['ul_northing'],
                           meta_dict['east_resolution'], meta_dict['north_resolution'],
                           meta_dict['img_width'], meta_dict['img_height'], propagate=True)

    # dsm = proj_to_utm_grid(np.hstack((lat, lon, alt)), meta_dict['ul_easting'], meta_dict['ul_northing'],
    #                        0.5, 0.5,
    #                        int(meta_dict['img_width']*0.3/0.5), int(meta_dict['img_height']*0.3/0.5))

    void_ratio = np.sum(np.isnan(dsm)) / dsm.size
    with open(os.path.join(out_dir, 'aggregate_3d_info.txt'), 'w') as fp:
        fp.write('completeness: {} %'.format((1.0-void_ratio) * 100.0))
    write_dsm_tif(dsm, meta_dict, os.path.join(out_dir, 'aggregate_3d.tif'))
    plot_height_map(dsm, os.path.join(out_dir, 'aggregate_3d.jpg'),
                    save_cbar=True, force_range=(meta_dict['alt_min'], meta_dict['alt_max']))

    # convert to utm coordinate frame
    # note the normals are in ENU system, not sure how to convert to UTM
    east, north = latlon_to_eastnorh(lat, lon)
    points = np.hstack((east, north, alt, points[:, 3:]))
    comment_1 = 'projection: UTM {}{}'.format(meta_dict['zone_number'], meta_dict['hemisphere'])
    comments = [comment_1,]
    np2ply(points, os.path.join(out_dir, 'aggregate_3d.ply'), comments)


if __name__ == '__main__':
    work_dir = '/data2/kz298/mvs3dm_result/MasterProvisional2'
    run_fuse(work_dir)
