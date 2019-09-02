from lib.run_cmd import run_cmd
import os
from lib.ply_np_converter import ply2np, np2ply
import json
# from lib.proj_to_utm_grid import proj_to_utm_grid
from coordinate_system import local_to_global
import numpy as np
from visualization.plot_height_map import plot_height_map
from lib.dsm_util import write_dsm_tif
from lib.latlon_utm_converter import latlon_to_eastnorh
from lib.proj_to_grid import proj_to_grid
import cv2


def fuse(colmap_dir):
    cmd = 'colmap stereo_fusion --workspace_path {colmap_dir}/mvs \
                         --output_path {colmap_dir}/mvs/fused.ply \
                         --input_type geometric \
                         --StereoFusion.min_num_pixels 4\
                         --StereoFusion.max_reproj_error 1\
                         --StereoFusion.max_depth_error 0.4\
                         --StereoFusion.max_normal_error 10'.format(colmap_dir=colmap_dir)
    run_cmd(cmd)


def run_fuse(work_dir):
    fuse(os.path.join(work_dir, 'colmap'))

    if not os.path.exists(os.path.join(work_dir, 'mvs_results')):
        os.mkdir(os.path.join(work_dir, 'mvs_results'))

    out_dir = os.path.join(work_dir, 'mvs_results/aggregate_3d')
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    points, color, comments = ply2np(os.path.join(work_dir, 'colmap/mvs/fused.ply'))

    lat, lon, alt = local_to_global(work_dir, points[:, 0:1], points[:, 1:2], points[:, 2:3])

    # convert to utm coordinate frame
    # note the normals are in ENU system, not sure how to convert to UTM
    east, north = latlon_to_eastnorh(lat, lon)
    points = np.hstack((east, north, alt))
    with open(os.path.join(work_dir, 'aoi.json')) as fp:
        aoi_dict = json.load(fp)
    comment_1 = 'projection: UTM {}{}'.format(aoi_dict['zone_number'], aoi_dict['hemisphere'])
    comments = [comment_1,]
    np2ply(points, os.path.join(out_dir, 'aggregate_3d.ply'), color=color, comments=comments, use_double=True)

    # write dsm to tif
    ul_e = aoi_dict['ul_easting']
    ul_n = aoi_dict['ul_northing']
    e_resolution = 0.5  # 0.5 meters per pixel
    n_resolution = 0.5 
    e_size = int(aoi_dict['width'] / e_resolution) + 1
    n_size = int(aoi_dict['height'] / n_resolution) + 1
    dsm = proj_to_grid(points, ul_e, ul_n, e_resolution, n_resolution, e_size, n_size, propagate=True)
    # median filter
    dsm = cv2.medianBlur(dsm.astype(np.float32), 3)
    write_dsm_tif(dsm, os.path.join(out_dir, 'aggregate_3d_dsm.tif'), 
                  (ul_e, ul_n, e_resolution, n_resolution), 
                  (aoi_dict['zone_number'], aoi_dict['hemisphere']), nodata_val=-10000)
    # create a preview file
    dsm = np.clip(dsm, aoi_dict['alt_min'], aoi_dict['alt_max'])
    plot_height_map(dsm, os.path.join(out_dir, 'aggregate_3d_dsm.jpg'), save_cbar=True)


if __name__ == '__main__':
    # work_dir = '/data2/kz298/mvs3dm_result/MasterProvisional2'
    # run_fuse(work_dir)

    from lib.dsm_util import read_dsm_tif
    ge_tif = '/bigdata/kz298/jacksonville_d4/buildings_dsm/dsm.tif'
    data, meta_dict = read_dsm_tif(ge_tif)

    work_dir = '/data2/kz298/core3d_result/aoi-d4-jacksonville'
    points = ply2np(os.path.join(work_dir, 'mvs_results/aggregate_3d/aggregate_3d.ply'))
    dsm = proj_to_grid(points[:, 0:3], meta_dict['ul_easting'], meta_dict['ul_northing'],
                           meta_dict['east_resolution'], meta_dict['north_resolution'],
                           meta_dict['img_width'], meta_dict['img_height'], propagate=True)

    out_dir = os.path.join(work_dir, 'texture')
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    np.save(os.path.join(out_dir, 'ge_dsm_utm.npy'), data)
    np.save(os.path.join(out_dir, 'our_dsm_utm.npy'), dsm)

    with open(os.path.join(out_dir, 'meta.json'), 'w') as fp:
        json.dump(meta_dict, fp)

    from visualization.plot_height_map import plot_height_map
    plot_height_map(dsm, os.path.join(out_dir, 'aggregate3d_dsm.png'))
