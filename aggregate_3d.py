#  ===============================================================================================================
#  Copyright (c) 2019, Cornell University. All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without modification, are permitted provided that
#  the following conditions are met:
#
#      * Redistributions of source code must retain the above copyright otice, this list of conditions and
#        the following disclaimer.
#
#      * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and
#        the following disclaimer in the documentation and/or other materials provided with the distribution.
#
#      * Neither the name of Cornell University nor the names of its contributors may be used to endorse or
#        promote products derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
#  WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
#  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE
#  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
#  TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
#  HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
#   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
#  OF SUCH DAMAGE.
#
#  Author: Kai Zhang (kz298@cornell.edu)
#
#  The research is based upon work supported by the Office of the Director of National Intelligence (ODNI),
#  Intelligence Advanced Research Projects Activity (IARPA), via DOI/IBC Contract Number D17PC00287.
#  The U.S. Government is authorized to reproduce and distribute copies of this work for Governmental purposes.
#  ===============================================================================================================


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
