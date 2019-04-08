from lib.run_cmd import run_cmd
import os
from lib.ply_np_converter import ply2np, np2ply
import json
from lib.proj_to_geo_grid import proj_to_geo_grid
from debugger.save_stats import save_stats


def fuse(colmap_dir):
    cmd = 'colmap stereo_fusion --workspace_path {colmap_dir}/mvs \
                         --output_path {colmap_dir}/mvs/fused.ply \
                         --input_type geometric \
                         --StereoFusion.min_num_pixels 3\
                         --StereoFusion.max_reproj_error 1\
                         --StereoFusion.max_depth_error 0.4\
                         --StereoFusion.max_normal_error 10'.format(colmap_dir=colmap_dir)
    run_cmd(cmd)


def run_fuse(tile_dir):
    fuse(os.path.join(tile_dir, 'colmap'))

    points = ply2np(os.path.join(tile_dir, 'colmap/mvs/fused.ply'))

    with open(os.path.join(tile_dir, 'ground_truth/dsm_gt_bbx_local.json')) as fp:
        bbx_local = json.load(fp)
    dsm = proj_to_geo_grid(points[:, 0:3], bbx_local['ul_easting'], bbx_local['ul_northing'],
                           bbx_local['resolution'], bbx_local['img_width'], bbx_local['img_height'])
    save_stats(dsm, 'colmap_fused_dsm', os.path.join(tile_dir, 'mvs_results'))

    # convert to utm coordinate frame
    with open(os.path.join(tile_dir, 'aoi.json')) as fp:
        aoi_dict = json.load(fp)
    aoi_ll_east = aoi_dict['ul_easting']
    aoi_ll_north = aoi_dict['ul_northing'] - aoi_dict['height']

    points[:, 0] += aoi_ll_east
    points[:, 1] += aoi_ll_north
    comment_1 = 'projection: UTM {}{}'.format(aoi_dict['zone_number'], aoi_dict['hemisphere'])
    comments = [comment_1,]
    np2ply(points, os.path.join(tile_dir, 'mvs_results/colmap_fused.ply'), comments)
