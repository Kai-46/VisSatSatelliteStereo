from lib.run_cmd import run_cmd


gpu_index = '0,1,2'


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
