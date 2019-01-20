import os
import json
from lib.clean_data import clean_data
from tile_cutter import TileCutter
from approx import Approx
import colmap_sfm_helper
import colmap_sfm
from lib.georegister_dense import georegister_dense
import shutil
import logging
from lib.run_cmd import run_cmd
from lib.timer import Timer
import numpy as np
from inspector.inspect_sfm import inspect_sfm
from lib.logger import GlobalLogger



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

        self.logger = GlobalLogger()

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
        self.logger.set_log_file(log_file)

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

    def run_derive_approx(self):
        work_dir = self.config['work_dir']

        # set log file to 'logs/log_derive_approx.txt'
        log_file = os.path.join(work_dir, 'logs/log_derive_approx.txt')
        self.logger.set_log_file(log_file)

        # create a local timer
        local_timer = Timer('Derive Approximation Module')
        local_timer.start()

        # derive approximations for later uses
        appr = Approx(work_dir)

        # pinhole_dict = appr.approx_pinhole_utm()
        # with open(os.path.join(work_dir, 'approx_pinhole_utm.json'), 'w') as fp:
        #     json.dump(pinhole_dict, fp, indent=2)

        perspective_dict = appr.approx_perspective_utm()
        with open(os.path.join(work_dir, 'approx_perspective_utm.json'), 'w') as fp:
            json.dump(perspective_dict, fp, indent=2)

        affine_dict = appr.approx_affine_latlon()
        with open(os.path.join(work_dir, 'approx_affine_latlon.json'), 'w') as fp:
            json.dump(affine_dict, fp, indent=2)

        # stop local timer
        local_timer.mark('Derive approximation done')
        logging.info(local_timer.summary())

    def run_colmap_sfm(self):
        work_dir = self.config['work_dir']
        colmap_dir = os.path.join(work_dir, 'colmap')

        # first time SfM
        log_file = os.path.join(work_dir, 'logs/log_sfm_perspective.txt')
        self.logger.set_log_file(log_file)

        # create a local timer
        local_timer = Timer('Colmap SfM Module, perspective camera')
        local_timer.start()

        colmap_sfm_helper.prep_for_sfm_perspective(work_dir, colmap_dir)
        colmap_sfm.run_sfm(os.path.join(colmap_dir, 'sfm_perspective'), 'PERSPECTIVE')

        # stop local timer
        local_timer.mark('Colmap SfM done')
        logging.info(local_timer.summary())

        # second time SfM
        log_file = os.path.join(work_dir, 'logs/log_sfm_pinhole.txt')
        self.logger.set_log_file(log_file)
        # create a local timer
        local_timer = Timer('Colmap SfM Module, pinhole camera')
        local_timer.start()

        colmap_sfm_helper.prep_for_sfm_pinhole(colmap_dir)
        colmap_sfm.run_sfm(os.path.join(colmap_dir, 'sfm_pinhole'), 'PINHOLE')

        # stop local timer
        local_timer.mark('Colmap SfM done')
        logging.info(local_timer.summary())

        # add inspector here
        inspect_sfm(colmap_dir)

    def run_colmap_mvs(self):
        work_dir = self.config['work_dir']

        # set log file
        log_file = os.path.join(work_dir, 'logs/log_mvs.txt')
        self.logger.set_log_file(log_file)
        # create a local timer
        local_timer = Timer('Colmap MVS Module')
        local_timer.start()

        # prepare dense reconstruction
        colmap_dir = os.path.join(work_dir, 'colmap')

        # prepare dense workspace
        cmd = 'colmap image_undistorter --max_image_size 5000 \
                            --image_path {colmap_dir}/sfm_pinhole/images  \
                            --input_path {colmap_dir}/sfm_pinhole/sparse_ba \
                            --output_path {colmap_dir}/mvs'.format(colmap_dir=colmap_dir)
        run_cmd(cmd)

        # PMVS
        cmd = 'colmap patch_match_stereo --workspace_path {colmap_dir}/mvs \
                        --PatchMatchStereo.window_radius 9 \
                        --PatchMatchStereo.filter_min_triangulation_angle 24.999 \
                        --PatchMatchStereo.geom_consistency 1 \
                        --PatchMatchStereo.filter_min_ncc 0.05 \
                        --PatchMatchStereo.gpu_index=1,2'.format(colmap_dir=colmap_dir)
        run_cmd(cmd)

        # stereo fusion
        cmd = 'colmap stereo_fusion --workspace_path {colmap_dir}/mvs \
                             --output_path {colmap_dir}/mvs/fused.ply \
                             --input_type geometric \
                             --StereoFusion.min_num_pixels 3'.format(colmap_dir=colmap_dir)
        run_cmd(cmd)

        # stop local timer
        local_timer.mark('Colmap MVS done')
        logging.info(local_timer.summary())


    def run_registration(self):
        work_dir = self.config['work_dir']
        colmap_dir = os.path.join(work_dir, 'colmap')

        # set log file to 'logs/log_derive_approx.txt'
        log_file = os.path.join(work_dir, 'logs/log_register.txt')
        self.logger.set_log_file(log_file)

        # create a local timer
        local_timer = Timer('Geo-registration Module')
        local_timer.start()

        # alignment
        from align_rpc import compute_transform
        M, t = compute_transform(work_dir)

        bbx = georegister_dense(os.path.join(colmap_dir, 'mvs/fused.ply'),
                          os.path.join(work_dir, 'register/registered_dense_points.ply'),
                          os.path.join(work_dir, 'aoi.json'), M, t, filter=True)

        with open(os.path.join(work_dir, 'register/registered_dense_points_bbx.json'), 'w') as fp:
            json.dump(bbx, fp)

        # stop local timer
        local_timer.mark('geo-registration done')
        logging.info(local_timer.summary())


    def run_evaluation(self):
        work_dir = self.config['work_dir']

        # set log file to 'logs/log_derive_approx.txt'
        log_file = os.path.join(work_dir, 'logs/log_evaluate.txt')
        self.logger.set_log_file(log_file)

        # create a local timer
        local_timer = Timer('Evaluation Module')
        local_timer.start()

        evaluate_dir = os.path.join(work_dir, 'evaluation')
        if not os.path.exists(evaluate_dir):
            os.mkdir(evaluate_dir)

        # flatten the point cloud
        # there's some problem here
        with open(os.path.join(work_dir, 'aoi.json')) as fp:
            aoi_dict = json.load(fp)

        # offset is the lower left
        ul_east = aoi_dict['x']
        ul_north = aoi_dict['y']
        resolution = 0.5
        width = int(1 + np.floor(aoi_dict['w'] / resolution))
        height = int(1 + np.floor(aoi_dict['h'] / resolution))

        #
        eval_point_cloud = '{evaluate_dir}/eval_point_cloud.ply'.format(evaluate_dir=evaluate_dir)
        eval_dsm = '{evaluate_dir}/eval_dsm.tif'.format(evaluate_dir=evaluate_dir)

        # copy point cloud to evaluation folder
        shutil.copy2(os.path.join(work_dir, 'register/registered_dense_points.ply'),
                     eval_point_cloud)

        # flatten point cloud
        cmd = '/home/cornell/kz298/s2p/bin/plyflatten {resolution} {eval_dsm} \
                            -srcwin "{xoff} {yoff} {xsize} {ysize}"'.format(resolution=resolution, eval_dsm=eval_dsm,
                            xoff=ul_east, yoff=ul_north, xsize=width, ysize=height)
        run_cmd(cmd, input=eval_point_cloud)

        # evaluate for core3d
        # cmd = 'python3 /home/cornell/kz298/core3d-metrics/core3dmetrics/run_geometrics.py --test-ignore 2 \
        #         -c {evaluate_config}'.format(evaluate_config=self.config['evaluate_config'])
        # run_cmd(cmd)

        # evaluate for mvs3dm
        ground_truth = self.config['ground_truth']

        eval_ground_truth = '{evaluate_dir}/eval_ground_truth.tif'.format(evaluate_dir=evaluate_dir)
        shutil.copy2(ground_truth, eval_ground_truth)

        cmd = '/data2/kz298/dataset/mvs3dm/Challenge_Data_and_Software/software/masterchallenge_metrics/build/bin/run-metrics \
              --cthreshold 1 \
              -t {} -i {}'.format(eval_ground_truth, eval_point_cloud)
        run_cmd(cmd)

        # stop local timer
        local_timer.mark('geo-registration done')
        logging.info(local_timer.summary())


if __name__ == '__main__':
    import sys
    config_file = sys.argv[1]

    pipeline = StereoPipeline(config_file)
    pipeline.run()
