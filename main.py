import os
import json
from lib.clean_data import clean_data
from tile_cutter import TileCutter
from approx import Approx
import colmap_sfm_perspective, colmap_sfm_pinhole
from correct_skew import remove_skew
import shutil
import logging
from lib.run_cmd import run_cmd
from lib.timer import Timer
import numpy as np
from lib.logger import GlobalLogger
from reparam_depth import reparam_depth
from colmap_mvs_commands import run_photometric_mvs, run_consistency_check
from lib.proj_to_geo_grid import proj_to_geo_grid
from lib.save_image_only import save_image_only
from process_dsm_gt import process_dsm_gt
from debugger.debug_perspective import debug_perspective
from debugger.convert_mvs_results import convert_depth_maps, convert_normal_maps
from debugger.aggregate_dsm import aggregate_dsm
import colmap_fuse
from debugger.merge_dsm import merge_dsm



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

        # write aoi json
        bbx_utm = self.config['bounding_box']
        aoi_dict = {'zone_number': bbx_utm['zone_number'],
                    'hemisphere': bbx_utm['hemisphere'],
                    'ul_easting': bbx_utm['ul_easting'],
                    'ul_northing': bbx_utm['ul_northing'],
                    'width': bbx_utm['width'],
                    'height': bbx_utm['height']}
        with open(os.path.join(self.config['work_dir'], 'aoi.json'), 'w') as fp:
            json.dump(aoi_dict, fp, indent=2)

        if self.config['steps_to_run']['process_dsm_gt']:
            process_dsm_gt(self.config['work_dir'], self.config['ground_truth'])

        if self.config['steps_to_run']['cut_image']:
            self.run_cut_image()

        if self.config['steps_to_run']['derive_approx']:
            self.run_derive_approx()

        if self.config['steps_to_run']['colmap_sfm']:
            self.run_colmap_sfm()

        if self.config['steps_to_run']['skew_correct']:
            self.run_skew_correct()

        if self.config['steps_to_run']['inspect_sfm']:
            self.run_inspect_sfm()

        if self.config['steps_to_run']['reparam_depth']:
            self.run_reparam_depth()

        if self.config['steps_to_run']['colmap_mvs']:
            self.run_colmap_mvs()

        if self.config['steps_to_run']['inspect_mvs']:
            self.run_inspect_mvs()

        if self.config['steps_to_run']['my_fuse']:
            self.run_my_fuse()

        if self.config['steps_to_run']['colmap_fuse']:
            self.run_colmap_fuse()

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
        cutter = TileCutter(cleaned_data_dir, work_dir, (self.config['z_min'], self.config['z_max']))
        cutter.cut_aoi(bbx['zone_number'], bbx['hemisphere'],
                       bbx['ul_easting'], bbx['ul_northing'], bbx['ul_easting'] + bbx['width'], bbx['ul_northing'] - bbx['height'])
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

        perspective_dict = appr.approx_perspective_utm()
        with open(os.path.join(work_dir, 'approx_perspective_utm.json'), 'w') as fp:
            json.dump(perspective_dict, fp, indent=2, sort_keys=True)

        affine_dict = appr.approx_affine_latlon()
        with open(os.path.join(work_dir, 'approx_affine_latlon.json'), 'w') as fp:
            json.dump(affine_dict, fp, indent=2, sort_keys=True)

        # stop local timer
        local_timer.mark('Derive approximation done')
        logging.info(local_timer.summary())

    def run_colmap_sfm(self, weight=1e-3):
        work_dir = os.path.abspath(self.config['work_dir'])
        colmap_dir = os.path.join(work_dir, 'colmap')
        subdirs = [
            colmap_dir,
            os.path.join(colmap_dir, 'sfm_perspective')
        ]

        for item in subdirs:
            if not os.path.exists(item):
                os.mkdir(item)

        log_file = os.path.join(work_dir, 'logs/log_sfm_perspective.txt')
        self.logger.set_log_file(log_file)
        # create a local timer
        local_timer = Timer('Colmap SfM Module, perspective camera')
        local_timer.start()

        sfm_dir = os.path.join(colmap_dir, 'sfm_perspective')
        # create a hard link to avoid copying of images
        if os.path.exists(os.path.join(colmap_dir, 'sfm_perspective/images')):
            os.unlink(os.path.join(colmap_dir, 'sfm_perspective/images'))
        os.symlink(os.path.join(work_dir, 'images'), os.path.join(colmap_dir, 'sfm_perspective/images'))
        init_camera_file = os.path.join(work_dir, 'approx_perspective_utm.json')
        colmap_sfm_perspective.run_sfm(work_dir, sfm_dir, init_camera_file, weight)

        # stop local timer
        local_timer.mark('Colmap SfM done')
        logging.info(local_timer.summary())

    def run_inspect_sfm(self):
        work_dir = os.path.abspath(self.config['work_dir'])
        sfm_dir = os.path.join(work_dir, 'colmap/sfm_perspective')

        log_file = os.path.join(work_dir, 'logs/log_inspect_sfm.txt')
        self.logger.set_log_file(log_file)
        local_timer = Timer('inspect sfm')
        local_timer.start()

        # inspect sfm perspective
        logging.info('inspecting sfm perspective...')
        colmap_sfm_perspective.check_sfm(work_dir, sfm_dir)
        debug_perspective(work_dir)

        # inspect sfm pinhole
        logging.info('inspecting sfm pinhole...')
        sfm_dir = os.path.join(work_dir, 'colmap/sfm_pinhole')
        warping_file = os.path.join(work_dir, 'colmap/skew_correct/affine_warpings.json')
        colmap_sfm_pinhole.check_sfm(work_dir, sfm_dir, warping_file)

        # stop local timer
        local_timer.mark('inspect sfm done')
        logging.info(local_timer.summary())

    def run_skew_correct(self):
        work_dir = os.path.abspath(self.config['work_dir'])
        colmap_dir = os.path.join(work_dir, 'colmap')
        subdirs = [
            colmap_dir,
            os.path.join(colmap_dir, 'skew_correct'),
            os.path.join(colmap_dir, 'sfm_pinhole')
        ]

        for item in subdirs:
            if not os.path.exists(item):
                os.mkdir(item)

        # second time SfM
        log_file = os.path.join(work_dir, 'logs/log_sfm_pinhole.txt')
        self.logger.set_log_file(log_file)
        # create a local timer
        local_timer = Timer('Colmap SfM Module, pinhole camera')
        local_timer.start()

        # skew-correct images
        perspective_img_dir = os.path.join(colmap_dir, 'sfm_perspective/images')
        # after bundle-adjustment
        perspective_file = os.path.join(colmap_dir, 'sfm_perspective/final_camera_dict.json')
        pinhole_img_dir = os.path.join(colmap_dir, 'skew_correct/images')
        pinhole_file = os.path.join(colmap_dir, 'skew_correct/pinhole_dict.json')
        warping_file = os.path.join(colmap_dir, 'skew_correct/affine_warpings.json')
        remove_skew(perspective_img_dir, perspective_file, pinhole_img_dir, pinhole_file, warping_file)

        # create a hard link to avoid copying of images
        sfm_dir = os.path.join(colmap_dir, 'sfm_pinhole')
        if os.path.exists(os.path.join(colmap_dir, 'sfm_pinhole/images')):
            os.unlink(os.path.join(colmap_dir, 'sfm_pinhole/images'))
        os.symlink(os.path.join(colmap_dir, 'skew_correct/images'), os.path.join(colmap_dir, 'sfm_pinhole/images'))
        init_camera_file = os.path.join(colmap_dir, 'skew_correct/pinhole_dict.json')

        colmap_sfm_pinhole.run_sfm(work_dir, sfm_dir, init_camera_file)

        # stop local timer
        local_timer.mark('Colmap SfM done')
        logging.info(local_timer.summary())

    def run_reparam_depth(self):
        work_dir = self.config['work_dir']

        # set log file
        log_file = os.path.join(work_dir, 'logs/log_reparam_depth.txt')
        self.logger.set_log_file(log_file)
        # create a local timer
        local_timer = Timer('reparametrize depth')
        local_timer.start()

        # prepare dense reconstruction
        colmap_dir = os.path.join(work_dir, 'colmap')

        mvs_dir = os.path.join(colmap_dir, 'mvs')
        if not os.path.exists(mvs_dir):
            os.mkdir(mvs_dir)

        # prepare dense workspace
        cmd = 'colmap image_undistorter --max_image_size 10000 \
                            --image_path {colmap_dir}/skew_correct/images  \
                            --input_path {colmap_dir}/sparse_for_mvs \
                            --output_path {colmap_dir}/mvs'.format(colmap_dir=colmap_dir)
        run_cmd(cmd)

        # compute depth ranges and generate last_rows.txt
        reparam_depth(os.path.join(mvs_dir, 'sparse'), mvs_dir)

    def run_colmap_mvs(self, window_radius=5):
        work_dir = self.config['work_dir']
        mvs_dir = os.path.join(work_dir, 'colmap/mvs')

        # set log file
        log_file = os.path.join(work_dir, 'logs/log_mvs.txt')
        self.logger.set_log_file(log_file)
        # create a local timer
        local_timer = Timer('Colmap MVS Module')
        local_timer.start()

        with open(os.path.join(mvs_dir, 'reparam_depth_range.txt')) as fp:
            tmp = fp.read().strip().split(' ')
            min_depth = float(tmp[0])
            max_depth = float(tmp[1])

        # first run PMVS without filtering
        run_photometric_mvs(mvs_dir, window_radius, depth_range=(min_depth, max_depth))

        # next do forward-backward checking and filtering
        run_consistency_check(mvs_dir, window_radius, depth_range=(min_depth, max_depth))

        # stop local timer
        local_timer.mark('Colmap MVS done')
        logging.info(local_timer.summary())

    def run_inspect_mvs(self):
        work_dir = self.config['work_dir']

        log_file = os.path.join(work_dir, 'logs/log_inspect_mvs.txt')
        self.logger.set_log_file(log_file)
        local_timer = Timer('inspect mvs')
        local_timer.start()

        logging.info('inspecting mvs ...')
        # add inspector
        type_name = 'geometric'
        convert_depth_maps(work_dir, depth_type=type_name)
        convert_normal_maps(work_dir, normal_type=type_name)

        aggregate_dsm(os.path.join(work_dir, 'mvs_results/height_maps/geo_grid_npy'),
                      os.path.join(work_dir, 'mvs_results/height_maps/geo_grid_aggregate'),
                      work_dir)
        # stop local timer
        local_timer.mark('inspect mvs done')
        logging.info(local_timer.summary())

    def run_colmap_fuse(self):
        work_dir = self.config['work_dir']
        # set log file
        log_file = os.path.join(work_dir, 'logs/log_colmap_fuse.txt')
        self.logger.set_log_file(log_file)
        # create a local timer
        local_timer = Timer('Colmap Fusion Module')
        local_timer.start()

        colmap_fuse.run_fuse(work_dir)

        # stop local timer
        local_timer.mark('Colmap Fusion done')
        logging.info(local_timer.summary())

    def run_my_fuse(self):
        work_dir = self.config['work_dir']
        # set log file
        log_file = os.path.join(work_dir, 'logs/log_my_fuse.txt')
        self.logger.set_log_file(log_file)
        # create a local timer
        local_timer = Timer('Fusion Module')
        local_timer.start()

        merge_dsm(work_dir)

        # stop local timer
        local_timer.mark('Fusion done')
        logging.info(local_timer.summary())

    def run_evaluation(self):
        work_dir = self.config['work_dir']
        evaluate_dir = os.path.join(work_dir, 'evaluation')
        if not os.path.exists(evaluate_dir):
            os.mkdir(evaluate_dir)

        # set log file
        log_file = os.path.join(work_dir, 'logs/log_evaluate.txt')
        self.logger.set_log_file(log_file)

        # create a local timer
        local_timer = Timer('Evaluation Module')
        local_timer.start()

        # copy ground truth to evaluation folder
        ground_truth = self.config['ground_truth']
        eval_ground_truth = '{evaluate_dir}/eval_ground_truth.tif'.format(evaluate_dir=evaluate_dir)
        shutil.copy2(ground_truth, eval_ground_truth)

        if self.config['evaluate_config']:
            from lib.image_util import write_image, read_image
            _, geo, proj, meta, width, height = read_image(eval_ground_truth)

            dsm = np.load(os.path.join(work_dir, 'mvs_results/merged_dsm.npy'))
            write_image(dsm, os.path.join(work_dir, 'evaluation/dsm.tif'), geo=geo, proj=proj, meta=meta, no_data=-9999)

            # evaluate for core3d
            # cmd = 'python3 /home/cornell/kz298/core3d-metrics/core3dmetrics/run_geometrics.py --test-ignore 2 \
            #         -c {}'.format(self.config['evaluate_config'])

            cmd = 'python3 /data2/kz298/core3d-metrics/core3dmetrics/run_geometrics.py --test-ignore 2 \
                    -c {}'.format(self.config['evaluate_config'])
            run_cmd(cmd)

            subdir = os.path.join(work_dir, 'evaluation/my_fuse')
            if os.path.exists(subdir):
                shutil.rmtree(subdir)
            if not os.path.exists(subdir):
                os.mkdir(subdir)
            for item in os.listdir(os.path.join(work_dir, 'evaluation')):
                if item == 'eval_ground_truth.tif':
                    continue

                shutil.move(os.path.join(work_dir, 'evaluation', item), subdir)

            dsm = np.load(os.path.join(work_dir, 'mvs_results/colmap_fused_dsm.npy'))
            write_image(dsm, os.path.join(work_dir, 'evaluation/dsm.tif'), geo=geo, proj=proj, meta=meta, no_data=-9999)

            # evaluate for core3d
            # cmd = 'python3 /home/cornell/kz298/core3d-metrics/core3dmetrics/run_geometrics.py --test-ignore 2 \
            #         -c {}'.format(self.config['evaluate_config'])
            #

            cmd = 'python3 /data2/kz298/core3d-metrics/core3dmetrics/run_geometrics.py --test-ignore 2 \
                    -c {}'.format(self.config['evaluate_config'])
            run_cmd(cmd)

        else:
            # analysis for mvs3dm
            from lib.image_util import write_image, read_image
            _, geo, proj, meta, width, height = read_image(eval_ground_truth)

            dsm = np.load(os.path.join(work_dir, 'mvs_results/merged_dsm.npy'))
            write_image(dsm, os.path.join(work_dir, 'mvs_results/merged_dsm.tif'), geo=geo, proj=proj, meta=meta, no_data=-9999)

            # align
            cmd = '/home/cornell/kz298/pubgeo/build/align3d {} {} maxt=10.0'.format(eval_ground_truth, os.path.join(work_dir, 'mvs_results/merged_dsm.tif'))
            run_cmd(cmd)

            # another dsm
            dsm = np.load(os.path.join(work_dir, 'mvs_results/colmap_fused_dsm.npy'))
            write_image(dsm, os.path.join(work_dir, 'mvs_results/colmap_fused_dsm.tif'), geo=geo, proj=proj, meta=meta, no_data=-9999)

            # align
            cmd = '/home/cornell/kz298/pubgeo/build/align3d {} {} maxt=10.0'.format(eval_ground_truth, os.path.join(work_dir, 'mvs_results/colmap_fused_dsm.tif'))
            run_cmd(cmd)

            # evaluate for mvs3dm
            logging.info('\n\nevaluating my fusion ...')
            eval_point_cloud = os.path.join(work_dir, 'mvs_results/merged_dsm.ply')
            cmd = '/bigdata/kz298/dataset/mvs3dm/Challenge_Data_and_Software/software/masterchallenge_metrics/build/bin/run-metrics \
                  --cthreshold 1 \
                  -t {} -i {}'.format(eval_ground_truth, eval_point_cloud)
            run_cmd(cmd)

            logging.info('\n\nevaluating colmap fusion ...')

            eval_point_cloud = os.path.join(work_dir, 'mvs_results/colmap_fused.ply')
            cmd = '/bigdata/kz298/dataset/mvs3dm/Challenge_Data_and_Software/software/masterchallenge_metrics/build/bin/run-metrics \
                  --cthreshold 1 \
                  -t {} -i {}'.format(eval_ground_truth, eval_point_cloud)
            run_cmd(cmd)

        # stop local timer
        local_timer.mark('geo-registration done')
        logging.info(local_timer.summary())

    def search_window_radius(self, window_radius_list):
        work_dir = self.config['work_dir']
        for window_radius in window_radius_list:
            print('current window radius: {}'.format(window_radius))
            self.run_colmap_mvs(window_radius)
            self.run_inspect_mvs()
            self.run_my_fuse()
            self.run_colmap_fuse()
            self.run_evaluation()
            os.rename(os.path.join(work_dir, 'mvs_results'),
                      os.path.join(work_dir, 'mvs_results_winrad_{}'.format(window_radius)))
            os.rename(os.path.join(work_dir, 'logs/log_mvs.txt'),
                      os.path.join(work_dir, 'logs/log_mvs_winrad_{}.txt'.format(window_radius)))
            os.rename(os.path.join(work_dir, 'logs/log_evaluate.txt'),
                      os.path.join(work_dir, 'logs/log_evaluate_winrad_{}.txt'.format(window_radius)))

    def search_regularization_weight(self, weight_list):
        work_dir = self.config['work_dir']
        for weight in weight_list:
            print('current weight: {}'.format(weight))
            self.run_colmap_sfm(weight=weight)
            self.run_skew_correct()
            self.run_reparam_depth()
            self.run_colmap_mvs(window_radius=5)
            self.run_inspect_mvs()
            self.run_my_fuse()
            self.run_colmap_fuse()
            self.run_evaluation()
            os.rename(os.path.join(work_dir, 'mvs_results'),
                      os.path.join(work_dir, 'mvs_results_weight_{:.8}'.format(weight)))
            os.rename(os.path.join(work_dir, 'logs/log_sfm_perspective.txt'),
                      os.path.join(work_dir, 'logs/log_sfm_perspective_weight_{:.8}.txt'.format(weight)))
            os.rename(os.path.join(work_dir, 'logs/log_sfm_pinhole.txt'),
                      os.path.join(work_dir, 'logs/log_sfm_pinhole_weight_{:.8}.txt'.format(weight)))
            os.rename(os.path.join(work_dir, 'logs/log_mvs.txt'),
                      os.path.join(work_dir, 'logs/log_mvs_weight_{:.8}.txt'.format(weight)))
            os.rename(os.path.join(work_dir, 'logs/log_evaluate.txt'),
                      os.path.join(work_dir, 'logs/log_evaluate_weight_{:.8}.txt'.format(weight)))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Satellite Stereo')
    parser.add_argument('--config_file', type=str,
                        help='configuration file')
    parser.add_argument('--search_window', type=int, default=0,
                        help='0 or 1, whether to search window radius')
    parser.add_argument('--search_weight', type=int, default=0,
                        help='0 or 1, whether to search regularization weight')
    parser.add_argument('--gpu_indices', type=str,
                        help='gpu devices to use')

    args = parser.parse_args()

    pipeline = StereoPipeline(args.config_file)

    if args.search_window > 0:
        pipeline.search_window_radius([2, 3, 5, 7, 9, 11, 13, 15, 17, 19])
    elif args.search_weight > 0:
        # pipeline.search_regularization_weight([0., 0.001, 0.01, 0.1, 1., 10., 100.])
        # pipeline.search_regularization_weight([1., 10., 100.])
        pipeline.search_regularization_weight([100., 1000.])
    else:
        pipeline.run()
