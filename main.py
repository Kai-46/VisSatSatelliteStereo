import os
import json
from lib.clean_data import clean_data
from tile_cutter import TileCutter
from approx import Approx
from prep_for_colmap import prep_for_sfm, prep_for_mvs, create_init_files
from lib.georegister_dense import georegister_dense
import shutil
import logging
from lib.run_cmd import run_cmd
from lib.timer import Timer


class StereoPipeline(object):
    def __init__(self, config_file):
        with open(config_file) as fp:
            self.config = json.load(fp)

        # make work_dir
        if not os.path.exists(self.config['work_dir']):
            os.mkdir(self.config['work_dir'])
        logs_subdir = os.path.join(self.config['work_dir'], 'logs')
        if not os.path.exists(logs_subdir):
            os.mkdir(logs_subdir)

    def run(self):
        print(self.config)

        if self.config['steps_to_run']['cut_image']:
            self.run_cut_image()

        if self.config['steps_to_run']['derive_approx']:
            self.run_derive_approx()

        if self.config['steps_to_run']['colmap_sfm']:
            self.run_colmap_sfm()

        if self.config['steps_to_run']['colmap_mvs']:
            self.run_colmap_mvs()

        if self.config['steps_to_run']['register']:
            self.run_registration()

        if self.config['steps_to_run']['evaluate']:
            self.run_evaluation()

    def run_cut_image(self):
        dataset_dir = self.config['dataset_dir']
        work_dir = self.config['work_dir']
        bbx = self.config['bounding_box']

        # set log file to 'logs/log_cut_image.txt'
        log_file = os.path.join(work_dir, 'logs/log_cut_image.txt')
        log_hanlder = logging.FileHandler(log_file, 'w')
        log_hanlder.setLevel(logging.INFO)
        logging.root.setLevel(logging.INFO)
        logging.root.addHandler(log_hanlder)

        # create a local timer
        local_timer = Timer('Cut Image Module')
        local_timer.start()

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
        # stop local timer
        local_timer.mark('cut image done')
        logging.info(local_timer.summary())

        # remove logging handler for later use
        logging.root.removeHandler(log_hanlder)

    def run_derive_approx(self):
        work_dir = self.config['work_dir']

        # set log file to 'logs/log_derive_approx.txt'
        log_file = os.path.join(work_dir, 'logs/log_derive_approx.txt')
        log_hanlder = logging.FileHandler(log_file, 'w')
        log_hanlder.setLevel(logging.INFO)
        logging.root.setLevel(logging.INFO)
        logging.root.addHandler(log_hanlder)

        # create a local timer
        local_timer = Timer('Derive Approximation Module')
        local_timer.start()

        # derive approximations for later uses
        appr = Approx(work_dir)
        perspective_dict = appr.approx_perspective_utm()
        with open(os.path.join(work_dir, 'approx_perspective_utm.json'), 'w') as fp:
            json.dump(perspective_dict, fp, indent=2)

        affine_dict = appr.approx_affine_latlon()
        with open(os.path.join(work_dir, 'approx_affine_latlon.json'), 'w') as fp:
            json.dump(affine_dict, fp, indent=2)

        # stop local timer
        local_timer.mark('Derive approximation done')
        logging.info(local_timer.summary())

        # remove logging handler for later use
        logging.root.removeHandler(log_hanlder)

    def run_colmap_sfm(self):
        work_dir = self.config['work_dir']
        # set log file to 'logs/log_derive_approx.txt'
        log_file = os.path.join(work_dir, 'logs/log_sfm.txt')
        log_hanlder = logging.FileHandler(log_file, 'w')
        log_hanlder.setLevel(logging.INFO)
        logging.root.setLevel(logging.INFO)
        logging.root.addHandler(log_hanlder)

        # create a local timer
        local_timer = Timer('Colmap SfM Module')
        local_timer.start()

        # prepare colmap workspace
        colmap_dir = os.path.join(work_dir, 'colmap')
        if not os.path.exists(colmap_dir):
            os.mkdir(colmap_dir)
        prep_for_sfm(work_dir, colmap_dir)

        # feature extraction
        cmd = 'colmap feature_extractor --database_path {colmap_dir}/database.db \
                                 --image_path {colmap_dir}/images/ \
                                --ImageReader.camera_model PERSPECTIVE \
                                --SiftExtraction.max_image_size 5000  \
                                --SiftExtraction.estimate_affine_shape 1 \
                                --SiftExtraction.domain_size_pooling 1'.format(colmap_dir=colmap_dir)
        run_cmd(cmd)

        # seems that we need to copy camera intrinsics into database
        # this is important??  Maybe not

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
                                         --Mapper.filter_min_tri_angle 1.5 \
                                         --Mapper.tri_min_angle 1.5 \
                                         --Mapper.filter_max_reproj_error 4 \
                                         --Mapper.max_extra_param 1e100 \
                                         --Mapper.ba_local_num_images 50 \
                                         --Mapper.ba_local_max_num_iterations 40 \
                                         --Mapper.ba_global_max_num_iterations 40'.format(colmap_dir=colmap_dir)
        run_cmd(cmd)

        # convert to txt format
        cmd = 'colmap model_converter --input_path {colmap_dir}/sparse  \
                                      --output_path {colmap_dir}/sparse \
                                      --output_type TXT'.format(colmap_dir=colmap_dir)
        run_cmd(cmd)

        # stop local timer
        local_timer.mark('Colmap SfM done')
        logging.info(local_timer.summary())

        # remove logging handler for later use
        logging.root.removeHandler(log_hanlder)

    def run_colmap_mvs(self):
        work_dir = self.config['work_dir']
        # set log file to 'logs/log_derive_approx.txt'
        log_file = os.path.join(work_dir, 'logs/log_mvs.txt')
        log_hanlder = logging.FileHandler(log_file, 'w')
        log_hanlder.setLevel(logging.INFO)
        logging.root.setLevel(logging.INFO)
        logging.root.addHandler(log_hanlder)

        # create a local timer
        local_timer = Timer('Colmap MVS Module')
        local_timer.start()

        # prepare dense reconstruction
        colmap_dir = os.path.join(work_dir, 'colmap')
        prep_for_mvs(colmap_dir)

        # cmd = 'colmap image_undistorter --max_image_size 5000 \
        #                     --image_path {colmap_dir}/images  \
        #                     --input_path {colmap_dir}/sparse \
        #                     --output_path {colmap_dir}/dense'.format(colmap_dir=colmap_dir)
        # run_cmd(cmd)

        # PMVS
        cmd = 'colmap patch_match_stereo --workspace_path {colmap_dir}/dense \
                        --PatchMatchStereo.window_radius 7 \
                        --PatchMatchStereo.filter_min_triangulation_angle 1 \
                        --PatchMatchStereo.geom_consistency 1 \
                        --PatchMatchStereo.filter_min_ncc 0.05'.format(colmap_dir=colmap_dir)
        run_cmd(cmd)

        # stereo fusion
        cmd = 'colmap stereo_fusion --workspace_path {colmap_dir}/dense \
                             --output_path {colmap_dir}/dense/fused.ply \
                             --input_type geometric \
                             --StereoFusion.min_num_pixels 3'.format(colmap_dir=colmap_dir)
        run_cmd(cmd)

        # stop local timer
        local_timer.mark('Colmap MVS done')
        logging.info(local_timer.summary())

        # remove logging handler for later use
        logging.root.removeHandler(log_hanlder)

    def run_registration(self):
        work_dir = self.config['work_dir']
        colmap_dir = os.path.join(work_dir, 'colmap')

        # set log file to 'logs/log_derive_approx.txt'
        log_file = os.path.join(work_dir, 'logs/log_register.txt')
        log_hanlder = logging.FileHandler(log_file, 'w')
        log_hanlder.setLevel(logging.INFO)
        logging.root.setLevel(logging.INFO)
        logging.root.addHandler(log_hanlder)

        # create a local timer
        local_timer = Timer('Geo-registration Module')
        local_timer.start()

        # alignment
        from align_rpc import compute_transform
        M, t = compute_transform(work_dir)

        georegister_dense(os.path.join(colmap_dir, 'dense/fused.ply'),
                          os.path.join(colmap_dir, 'dense/fused_registered.ply'),
                          os.path.join(work_dir, 'aoi.json'), M, t)

        # stop local timer
        local_timer.mark('geo-registration done')
        logging.info(local_timer.summary())

        # remove logging handler for later use
        logging.root.removeHandler(log_hanlder)

    def run_evaluation(self):
        work_dir = self.config['work_dir']
        colmap_dir = os.path.join(work_dir, 'colmap')

        # set log file to 'logs/log_derive_approx.txt'
        log_file = os.path.join(work_dir, 'logs/log_evaluate.txt')
        log_hanlder = logging.FileHandler(log_file, 'w')
        log_hanlder.setLevel(logging.INFO)
        logging.root.setLevel(logging.INFO)
        logging.root.addHandler(log_hanlder)

        # create a local timer
        local_timer = Timer('Evaluation Module')
        local_timer.start()

        evaluate_dir = os.path.join(work_dir, 'evaluation')
        if not os.path.exists(evaluate_dir):
            os.mkdir(evaluate_dir)

        # flatten the point cloud
        cmd = 'echo {colmap_dir}/dense/fused_registered.ply | \
            /home/cornell/kz298/s2p/bin/plyflatten 0.5 {evaluate_dir}/dsm.tif'.format(colmap_dir=colmap_dir, evaluate_dir=evaluate_dir)
        run_cmd(cmd)

        # evaluate
        cmd = 'python3 /home/cornell/kz298/core3d-metrics/core3dmetrics/run_geometrics.py --test-ignore 2 \
                -c {evaluate_config}'.format(evaluate_config=self.config['evaluate_config'])
        run_cmd(cmd)

        # stop local timer
        local_timer.mark('geo-registration done')
        logging.info(local_timer.summary())

        # remove logging handler for later use
        logging.root.removeHandler(log_hanlder)

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

    pipeline = StereoPipeline(config_file)
    pipeline.run()
