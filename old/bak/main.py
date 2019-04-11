import os
import json
from clean_data import clean_data
from tile_cutter import TileCutter
from approx import Approx
# from colmap_sfm_helper import prep_for_sfm, prep_for_mvs, create_init_files
from lib.georegister_dense import georegister_dense
import shutil
import logging
from lib.run_cmd import run_cmd
from lib.timer import Timer


def sfm():
    pass

def mvs():
    pass


def main(config_file):
    timer = Timer('Satellite Stereo')
    timer.start()

    with open(config_file) as fp:
        config = json.load(fp)

    dataset_dir = config['dataset_dir']
    work_dir = config['work_dir']
    bbx = config['bounding_box']

    # create work_dir
    if not os.path.exists(work_dir):
        os.mkdir(work_dir)

    # set log file
    log_file = os.path.join(work_dir, 'log.txt')
    logging.basicConfig(filename=log_file, level=logging.INFO, filemode='w')

    # write config_file
    logging.info(config_file)

    # clean data
    cleaned_data_dir = os.path.join(work_dir, 'cleaned_data')
    if os.path.exists(cleaned_data_dir):  # remove cleaned_data_dir
        shutil.rmtree(cleaned_data_dir, ignore_errors=True)
    os.mkdir(cleaned_data_dir)

    clean_data(dataset_dir, cleaned_data_dir)

    # cut image and tone map
    cutter = TileCutter(cleaned_data_dir, work_dir)
    cutter.cut_aoi(bbx['zone_number'], bbx['zone_letter'],
                   bbx['x'], bbx['y'], bbx['x'] + bbx['w'], bbx['y'] - bbx['h'])

    # record time
    now, since_last, since_start = timer.mark('till cut image done')
    logging.info('\nelapsed: {} min\n'.format(since_last))

    # derive approximations for later uses
    appr = Approx(work_dir)
    perspective_dict = appr.approx_perspective_utm()
    with open(os.path.join(work_dir, 'approx_perspective_utm.json'), 'w') as fp:
        json.dump(perspective_dict, fp, indent=2)

    affine_dict = appr.approx_affine_latlon()
    with open(os.path.join(work_dir, 'approx_affine_latlon.json'), 'w') as fp:
        json.dump(affine_dict, fp, indent=2)

    # # prepare colmap workspace
    colmap_dir = os.path.join(work_dir, 'colmap')
    if not os.path.exists(colmap_dir):
        os.mkdir(colmap_dir)
    prep_for_sfm(work_dir, colmap_dir)

    # record time
    now, since_last, since_start = timer.mark('till pre-processing done')
    logging.info('\nelapsed: {} min\n'.format(since_last))

    # start colmap commands
    # feature extraction
    cmd = 'colmap feature_extractor --database_path {colmap_dir}/database.db \
                             --image_path {colmap_dir}/images/ \
                            --ImageReader.camera_model PINHOLE \
                            --SiftExtraction.max_image_size 5000  \
                            --SiftExtraction.estimate_affine_shape 1 \
                            --SiftExtraction.domain_size_pooling 1'.format(colmap_dir=colmap_dir)
    run_cmd(cmd)

    # feature matching
    cmd = 'colmap exhaustive_matcher --database_path {colmap_dir}/database.db \
                                    --SiftMatching.guided_matching 1'.format(colmap_dir=colmap_dir)

    run_cmd(cmd)

    # create initial poses
    create_init_files(colmap_dir)

    # triangulate points
    cmd = 'colmap point_triangulator --Mapper.ba_refine_principal_point 1 \
                                     --database_path {colmap_dir}/database.db \
                                     --image_path {colmap_dir}/images/ \
                                     --input_path {colmap_dir}/init \
                                     --output_path {colmap_dir}/sparse \
                                     --Mapper.filter_min_tri_angle 2 \
                                     --Mapper.tri_min_angle 2 \
                                     --Mapper.max_extra_param 1.7976931348623157e+308 \
                                     --Mapper.ba_local_max_num_iterations 40 \
                                     --Mapper.ba_local_max_refinements 3 \
                                     --Mapper.ba_global_max_num_iterations 100'.format(colmap_dir=colmap_dir)
    run_cmd(cmd)

    # global bundle adjustment
    cmd = 'colmap bundle_adjuster --input_path {colmap_dir}/sparse --output_path {colmap_dir}/sparse_ba \
    	                            --BundleAdjustment.max_num_iterations 1000 \
    	                            --BundleAdjustment.refine_principal_point 1 \
    	                            --BundleAdjustment.function_tolerance 1e-6 \
    	                            --BundleAdjustment.gradient_tolerance 1e-10 \
    	                            --BundleAdjustment.parameter_tolerance 1e-8'.format(colmap_dir=colmap_dir)
    run_cmd(cmd)

    # record time
    now, since_last, since_start = timer.mark('till sfm done')
    logging.info('\nelapsed: {} min\n'.format(since_last))

    # prepare dense reconstruction
    prep_for_mvs(colmap_dir)

    # cmd = 'colmap image_undistorter --max_image_size 5000 \
    #                     --image_path {colmap_dir}/images  \
    #                     --input_path {colmap_dir}/sparse_ba \
    #                     --output_path {colmap_dir}/dense'.format(colmap_dir=colmap_dir)
    # run_cmd(cmd)

    # PMVS
    cmd = 'colmap patch_match_stereo --workspace_path {colmap_dir}/dense \
                    --PatchMatchStereo.window_radius 9 \
                    --PatchMatchStereo.filter_min_triangulation_angle 1 \
                    --PatchMatchStereo.geom_consistency 1 \
                    --PatchMatchStereo.filter_min_ncc 0.05'.format(colmap_dir=colmap_dir)
    run_cmd(cmd)

    # record time
    now, since_last, since_start = timer.mark('till mvs done')
    logging.info('\nelapsed: {} min\n'.format(since_last))

    # stereo fusion
    cmd = 'colmap stereo_fusion --workspace_path {colmap_dir}/dense \
                         --output_path {colmap_dir}/dense/fused.ply \
                         --input_type geometric \
                         --StereoFusion.min_num_pixels 3'.format(colmap_dir=colmap_dir)
    run_cmd(cmd)

    # record time
    now, since_last, since_start = timer.mark('till fusion done')
    logging.info('\nelapsed: {} min\n'.format(since_last))

    # alignment
    # decide which solution to take
    from old.align_sparse import compute_transform
    # from align_rpc import compute_transform
    # from align_cam import compute_transform
    c, R, t = compute_transform(work_dir)

    georegister_dense(os.path.join(colmap_dir, 'dense/fused.ply'),
                      os.path.join(colmap_dir, 'dense/fused_registered.ply'),
                      os.path.join(work_dir, 'aoi.json'), c, R, t)

    # record time
    now, since_last, since_start = timer.mark('till registration done')
    logging.info('\nelapsed: {} min\n'.format(since_last))

    # generate time consumption report
    logging.info(timer.summary())


if __name__ == '__main__':
    import sys
    config_file = sys.argv[1]

    # read config file
    #config_file = 'aoi_config/aoi-d1-wpafb.json'
    #config_file = 'aoi_config/aoi-d4-jacksonville.json'

    # config_files = ['aoi_config/aoi-d1-wpafb.json',
    #                 'aoi_config/aoi-d2-wpafb.json',
    #                 'aoi_config/aoi-d3-ucsd.json',
    #                 'aoi_config/aoi-d4-jacksonville.json']
    # for config_file in config_files:
    #     # remove all the existing logging handlers
    #     for handler in logging.root.handlers:
    #         logging.root.removeHandler(handler)
    #
    #     main(config_file)
    main(config_file)
