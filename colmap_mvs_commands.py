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


gpu_index = '-1'


def run_photometric_mvs(mvs_dir, window_radius, depth_range=None):
    cmd = 'colmap patch_match_stereo --workspace_path {} \
                    --PatchMatchStereo.window_radius {}\
                    --PatchMatchStereo.min_triangulation_angle 10.0 \
                    --PatchMatchStereo.filter 0 \
                    --PatchMatchStereo.geom_consistency 0 \
                    --PatchMatchStereo.gpu_index={} \
                    --PatchMatchStereo.num_samples 10 \
                    --PatchMatchStereo.num_iterations 12 \
                    --PatchMatchStereo.overwrite 1'.format(mvs_dir,
                                                           window_radius, gpu_index)
    if depth_range is not None:
        depth_min, depth_max = depth_range
        other_opts = '--PatchMatchStereo.depth_min {} \
                      --PatchMatchStereo.depth_max {}'.format(depth_min, depth_max)
        cmd = '{} {}'.format(cmd, other_opts)
    run_cmd(cmd)


def run_consistency_check(mvs_dir, window_radius, depth_range=None):
    # do forward-backward checking and filtering
    cmd = 'colmap patch_match_stereo --workspace_path {} \
                    --PatchMatchStereo.window_radius {} \
                    --PatchMatchStereo.min_triangulation_angle 10.0 \
                    --PatchMatchStereo.geom_consistency 1 \
                    --PatchMatchStereo.use_exist_photom 1 \
                    --PatchMatchStereo.overwrite 1 \
                    --PatchMatchStereo.geom_consistency_regularizer 100.0 \
                    --PatchMatchStereo.geom_consistency_max_cost 3 \
                    --PatchMatchStereo.filter 1 \
                    --PatchMatchStereo.filter_min_triangulation_angle 9.999 \
                    --PatchMatchStereo.filter_min_ncc -0.999 \
                    --PatchMatchStereo.filter_geom_consistency_max_cost 1 \
                    --PatchMatchStereo.filter_min_num_consistent 2 \
                    --PatchMatchStereo.gpu_index={} \
                    --PatchMatchStereo.num_samples 10 \
                    --PatchMatchStereo.num_iterations 1'.format(mvs_dir,
                                                                window_radius, gpu_index)
    if depth_range is not None:
        depth_min, depth_max = depth_range
        other_opts = '--PatchMatchStereo.depth_min {} \
                      --PatchMatchStereo.depth_max {}'.format(depth_min, depth_max)
        cmd = '{} {}'.format(cmd, other_opts)
    run_cmd(cmd)
