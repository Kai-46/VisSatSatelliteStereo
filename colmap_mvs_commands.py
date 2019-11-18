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


gpu_index = '-1'


def run_photometric_mvs(mvs_dir, window_radius, depth_range=None):
    cmd = 'colmap patch_match_stereo --workspace_path {} \
                    --PatchMatchStereo.window_radius {}\
                    --PatchMatchStereo.min_triangulation_angle 5.0 \
                    --PatchMatchStereo.filter 0 \
                    --PatchMatchStereo.geom_consistency 0 \
                    --PatchMatchStereo.gpu_index={} \
                    --PatchMatchStereo.num_samples 15 \
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
                    --PatchMatchStereo.min_triangulation_angle 5.0 \
                    --PatchMatchStereo.geom_consistency 1 \
                    --PatchMatchStereo.use_exist_photom 1 \
                    --PatchMatchStereo.overwrite 1 \
                    --PatchMatchStereo.geom_consistency_regularizer 0.3 \
                    --PatchMatchStereo.geom_consistency_max_cost 3 \
                    --PatchMatchStereo.filter 1 \
                    --PatchMatchStereo.filter_min_triangulation_angle 4.99 \
                    --PatchMatchStereo.filter_min_ncc 0.1 \
                    --PatchMatchStereo.filter_geom_consistency_max_cost 1 \
                    --PatchMatchStereo.filter_min_num_consistent 2 \
                    --PatchMatchStereo.gpu_index={} \
                    --PatchMatchStereo.num_samples 15 \
                    --PatchMatchStereo.num_iterations 1'.format(mvs_dir,
                                                                window_radius, gpu_index)
    if depth_range is not None:
        depth_min, depth_max = depth_range
        other_opts = '--PatchMatchStereo.depth_min {} \
                      --PatchMatchStereo.depth_max {}'.format(depth_min, depth_max)
        cmd = '{} {}'.format(cmd, other_opts)
    run_cmd(cmd)
