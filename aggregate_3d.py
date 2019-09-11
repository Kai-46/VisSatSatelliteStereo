from lib.run_cmd import run_cmd
import os
from lib.ply_np_converter import ply2np, np2ply
import json
from coordinate_system import local_to_global
import numpy as np
from lib.latlon_utm_converter import latlon_to_eastnorh
from produce_dsm import produce_dsm_from_points
import cv2

# the unit of max_depth_error is now in meter
def fuse(colmap_dir):
    cmd = 'colmap stereo_fusion --workspace_path {colmap_dir}/mvs \
                         --output_path {colmap_dir}/mvs/fused.ply \
                         --input_type geometric \
                         --StereoFusion.min_num_pixels 4\
                         --StereoFusion.max_reproj_error 2\
                         --StereoFusion.max_depth_error 1.0\
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
    tif_to_write = os.path.join(out_dir, 'aggregate_3d_dsm.tif')
    jpg_to_write = os.path.join(out_dir, 'aggregate_3d_dsm.jpg')
    produce_dsm_from_points(work_dir, points, tif_to_write, jpg_to_write)


if __name__ == '__main__':
    pass
