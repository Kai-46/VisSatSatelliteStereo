# ===============================================================================================================
#  This file is part of Creation of Operationally Realistic 3D Environment (CORE3D).                            =
#  Copyright 2019 Cornell University - All Rights Reserved                                                      =
#  -                                                                                                            =
#  NOTICE: All information contained herein is, and remains the property of General Electric Company            =
#  and its suppliers, if any. The intellectual and technical concepts contained herein are proprietary          =
#  to General Electric Company and its suppliers and may be covered by U.S. and Foreign Patents, patents        =
#  in process, and are protected by trade secret or copyright law. Dissemination of this information or         =
#  reproduction of this material is strictly forbidden unless prior written permission is obtained              =
#  from General Electric Company.                                                                               =
#  -                                                                                                            =
#  The research is based upon work supported by the Office of the Director of National Intelligence (ODNI),     =
#  Intelligence Advanced Research Projects Activity (IARPA), via DOI/IBC Contract Number D17PC00287.            =
#  The U.S. Government is authorized to reproduce and distribute copies of this work for Governmental purposes. =
# ===============================================================================================================

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

    if not os.path.exists(os.path.join(work_dir, 'mvs_results')):
        os.mkdir(os.path.join(work_dir, 'mvs_results'))
        
    out_dir = os.path.join(work_dir, 'mvs_results/aggregate_3d')
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    points = ply2np(os.path.join(work_dir, 'colmap/mvs/fused.ply'))
    lat, lon, alt = local_to_global(work_dir, points[:, 0:1], points[:, 1:2], points[:, 2:3])

    # convert to utm coordinate frame
    # note the normals are in ENU system, not sure how to convert to UTM
    east, north = latlon_to_eastnorh(lat, lon)
    with open(os.path.join(work_dir, 'aoi.json')) as fp:
        aoi_dict = json.load(fp)
    comment_1 = 'projection: UTM {}{}'.format(aoi_dict['zone_number'], aoi_dict['hemisphere'])
    comments = [comment_1,]
    np2ply(points, os.path.join(out_dir, 'aggregate_3d.ply'), comments)


if __name__ == '__main__':
    work_dir = '/data2/kz298/mvs3dm_result/MasterProvisional2'
    run_fuse(work_dir)
