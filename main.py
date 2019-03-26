import os
import json
from lib.clean_data import clean_data
from tile_cutter import TileCutter
from approx import Approx
import colmap_sfm_perspective, colmap_sfm_pinhole
from correct_skew import remove_skew
from lib.georegister_dense import georegister_dense
import shutil
import logging
from lib.run_cmd import run_cmd
from lib.timer import Timer
import numpy as np
from lib.logger import GlobalLogger
from reparam_depth import reparam_depth


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

        if self.config['steps_to_run']['colmap_fuse']:
            self.run_colmap_fuse()

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

        # pinhole_dict = appr.approx_pinhole_utm()
        # with open(os.path.join(work_dir, 'approx_pinhole_utm.json'), 'w') as fp:
        #     json.dump(pinhole_dict, fp, indent=2)

        perspective_dict = appr.approx_perspective_utm()
        with open(os.path.join(work_dir, 'approx_perspective_utm.json'), 'w') as fp:
            json.dump(perspective_dict, fp, indent=2, sort_keys=True)

        affine_dict = appr.approx_affine_latlon()
        with open(os.path.join(work_dir, 'approx_affine_latlon.json'), 'w') as fp:
            json.dump(affine_dict, fp, indent=2, sort_keys=True)

        # stop local timer
        local_timer.mark('Derive approximation done')
        logging.info(local_timer.summary())

    def run_colmap_sfm(self):
        work_dir = os.path.abspath(self.config['work_dir'])
        colmap_dir = os.path.join(work_dir, 'colmap')
        subdirs = [
            colmap_dir,
            os.path.join(colmap_dir, 'sfm_perspective'),
            os.path.join(colmap_dir, 'skew_correct'),
            os.path.join(colmap_dir, 'sfm_pinhole'),
            os.path.join(colmap_dir, 'mvs')
        ]

        for item in subdirs:
            if not os.path.exists(item):
                os.mkdir(item)

        # first time SfM
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
        colmap_sfm_perspective.run_sfm(work_dir, sfm_dir, init_camera_file)

        # add inspector
        colmap_sfm_perspective.check_sfm(work_dir, sfm_dir)

        # stop local timer
        local_timer.mark('Colmap SfM done')
        logging.info(local_timer.summary())

        # second time SfM
        log_file = os.path.join(work_dir, 'logs/log_sfm_pinhole.txt')
        self.logger.set_log_file(log_file)
        # create a local timer
        local_timer = Timer('Colmap SfM Module, pinhole camera')
        local_timer.start()

        # skew-correct images
        perspective_img_dir = os.path.join(colmap_dir, 'sfm_perspective/images')
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

        # add inspector
        colmap_sfm_pinhole.check_sfm(work_dir, sfm_dir, warping_file)

        # stop local timer
        local_timer.mark('Colmap SfM done')
        logging.info(local_timer.summary())


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

        # need to delete mvs
        mvs_dir = os.path.join(colmap_dir, 'mvs')
        for subdir in [os.path.join(mvs_dir, 'images'),
                       os.path.join(mvs_dir, 'sparse'),
                       os.path.join(mvs_dir, 'stereo')]:
            if os.path.exists(subdir):
                shutil.rmtree(subdir, ignore_errors=True)

        if not os.path.exists(mvs_dir):
            os.mkdir(mvs_dir)

        # prepare dense workspace
        cmd = 'colmap image_undistorter --max_image_size 5000 \
                            --image_path {colmap_dir}/skew_correct/images  \
                            --input_path {colmap_dir}/sparse_for_mvs \
                            --output_path {colmap_dir}/mvs'.format(colmap_dir=colmap_dir)
        run_cmd(cmd)

        # compute depth ranges and generate last_rows.txt
        reparam_depth(os.path.join(mvs_dir, 'sparse'), mvs_dir)

        win_radius = 2

        # first run PMVS without filtering
        gpu_index = '1,2'
        cmd = 'colmap patch_match_stereo --workspace_path {colmap_dir}/mvs \
                        --PatchMatchStereo.depth_min 0.008 \
                        --PatchMatchStereo.depth_max 0.098 \
                        --PatchMatchStereo.window_radius {win_radius}\
                        --PatchMatchStereo.min_triangulation_angle 10.0 \
                        --PatchMatchStereo.filter 0 \
                        --PatchMatchStereo.geom_consistency 0 \
                        --PatchMatchStereo.gpu_index={gpu_index} \
                        --PatchMatchStereo.num_samples 10 \
                        --PatchMatchStereo.num_iterations 12 \
                        --PatchMatchStereo.overwrite 1'.format(win_radius=win_radius,
                                                               colmap_dir=colmap_dir, gpu_index=gpu_index)
        run_cmd(cmd)

        # next do forward-backward checking and filtering
        cmd = 'colmap patch_match_stereo --workspace_path {colmap_dir}/mvs \
                        --PatchMatchStereo.depth_min 0.008 \
                        --PatchMatchStereo.depth_max 0.098 \
                        --PatchMatchStereo.window_radius {win_radius} \
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
                        --PatchMatchStereo.gpu_index={gpu_index} \
                        --PatchMatchStereo.num_samples 10 \
                        --PatchMatchStereo.num_iterations 1'.format(win_radius=win_radius,
                                                                    colmap_dir=colmap_dir, gpu_index=gpu_index)
        # for debugging
        run_cmd(cmd)

        # add inspector
        from convert_depth_maps import convert_depth_maps, convert_normal_maps
        convert_depth_maps(mvs_dir, os.path.join(mvs_dir, 'height_maps'), depth_type='geometric')
        convert_normal_maps(mvs_dir, os.path.join(mvs_dir, 'height_maps'), normal_type='geometric')

        # stop local timer
        local_timer.mark('Colmap MVS done')
        logging.info(local_timer.summary())

    def run_colmap_fuse(self):
        work_dir = self.config['work_dir']
        # prepare dense reconstruction
        colmap_dir = os.path.join(work_dir, 'colmap')

        # set log file
        log_file = os.path.join(work_dir, 'logs/log_fuse.txt')
        self.logger.set_log_file(log_file)
        # create a local timer
        local_timer = Timer('Colmap Fusion Module')
        local_timer.start()

        # stereo fusion
        # colmap's estimation of surface normal is very inaccurate
        # therefore we completely ignore the surface normal during fusion
        cmd = 'colmap stereo_fusion --workspace_path {colmap_dir}/mvs \
                             --output_path {colmap_dir}/mvs/fused.ply \
                             --input_type geometric \
                             --StereoFusion.min_num_pixels 2\
                             --StereoFusion.max_reproj_error 1\
                             --StereoFusion.max_depth_error 0.3\
                             --StereoFusion.max_normal_error 10'.format(colmap_dir=colmap_dir)
        run_cmd(cmd)

        # stop local timer
        local_timer.mark('Colmap Fusion done')
        logging.info(local_timer.summary())

    def run_registration(self):
        work_dir = self.config['work_dir']
        colmap_dir = os.path.join(work_dir, 'colmap')
        register_dir = os.path.join(work_dir, 'register')
        if not os.path.exists(register_dir):
            os.mkdir(register_dir)

        # set log file to 'logs/log_derive_approx.txt'
        log_file = os.path.join(work_dir, 'logs/log_register.txt')
        self.logger.set_log_file(log_file)

        # create a local timer
        local_timer = Timer('Geo-registration Module')
        local_timer.start()

        # alignment
        # from align_rpc import compute_transform
        # M, t = compute_transform(work_dir)
        # with open(os.path.join(colmap_dir, 'normalize.txt')) as fp:
        #     lines = fp.readlines()
        #     translation = [float(x) for x in lines[1].strip().split(' ')]
        #     scale = float(lines[3].strip())
        # M = np.identity(3) / scale
        # t = np.array(translation).reshape((1, 3))

        M = np.identity(3)
        t = np.zeros((1, 3))

        # add global shift
        with open(os.path.join(work_dir, 'aoi.json')) as fp:
            aoi_dict = json.load(fp)
        aoi_ll_east = aoi_dict['ul_easting']
        aoi_ll_north = aoi_dict['ul_northing'] - aoi_dict['height']
        t[0, 0] += aoi_ll_east
        t[0, 1] += aoi_ll_north

        bbx = georegister_dense(os.path.join(colmap_dir, 'mvs/fused.ply'),
                          os.path.join(register_dir, 'registered_dense_points.ply'),
                          os.path.join(work_dir, 'aoi.json'), M, t, filter=True)

        with open(os.path.join(register_dir, 'registered_dense_points_bbx.json'), 'w') as fp:
            json.dump(bbx, fp)

        # stop local timer
        local_timer.mark('geo-registration done')
        logging.info(local_timer.summary())

    def run_evaluation(self):
        work_dir = self.config['work_dir']
        evaluate_dir = os.path.join(work_dir, 'evaluation')
        if not os.path.exists(evaluate_dir):
            os.mkdir(evaluate_dir)

        # set log file to 'logs/log_derive_approx.txt'
        log_file = os.path.join(work_dir, 'logs/log_evaluate.txt')
        self.logger.set_log_file(log_file)

        # create a local timer
        local_timer = Timer('Evaluation Module')
        local_timer.start()

        ground_truth = self.config['ground_truth']
        eval_ground_truth = '{evaluate_dir}/eval_ground_truth.tif'.format(evaluate_dir=evaluate_dir)
        shutil.copy2(ground_truth, eval_ground_truth)

        # create a low-resolution ground-truth
        cmd = 'gdal_translate -tr 0.5 0.5 {evaluate_dir}/eval_ground_truth.tif {evaluate_dir}/eval_ground_truth_0.5.tif'.format(evaluate_dir=evaluate_dir)
        run_cmd(cmd)

        # normalize as png
        cmd = 'gdal_translate -ot Byte -tr 0.5 0.5 -of png -scale 15 32 0 255 {evaluate_dir}/eval_ground_truth.tif {evaluate_dir}/eval_ground_truth.tif.png'.format(evaluate_dir=evaluate_dir)
        run_cmd(cmd)

        # get the covered area the ground truth
        import subprocess, json
        process = subprocess.Popen(['gdalinfo', '-json', eval_ground_truth], stdout=subprocess.PIPE)
        out, err = process.communicate()
        meta = json.loads(out)

        aoi_ul_east, aoi_ul_north = meta['cornerCoordinates']['upperLeft']
        aoi_lr_east, aoi_lr_north = meta['cornerCoordinates']['lowerRight']
        aoi_width = aoi_lr_east - aoi_ul_east
        aoi_height = aoi_ul_north - aoi_lr_north

        # flatten the point cloud
        # there's some problem here
        for resolution in [0.3, 0.5]:
            width = int(np.floor(aoi_width / resolution))
            height = int(1+np.floor(aoi_height / resolution))

            #
            eval_point_cloud = '{evaluate_dir}/eval_point_cloud.ply'.format(evaluate_dir=evaluate_dir)
            eval_dsm = '{evaluate_dir}/eval_dsm_{resolution}.tif'.format(evaluate_dir=evaluate_dir, resolution=resolution)

            # copy point cloud to evaluation folder
            # shutil.copy2(os.path.join(work_dir, 'register/registered_dense_points.ply'),
            #              eval_point_cloud)

            # flatten point cloud
            cmd = '/home/cornell/kz298/s2p/bin/plyflatten {resolution} {eval_dsm} \
                                -srcwin "{xoff} {yoff} {xsize} {ysize}"'.format(resolution=resolution, eval_dsm=eval_dsm,
                                xoff=aoi_ul_east, yoff=aoi_ul_north, xsize=width, ysize=height)
            run_cmd(cmd, input=eval_point_cloud)

            # normalize as png
            cmd = 'gdal_translate -ot Byte -of png -scale 15 32 0 255 {eval_dsm} {eval_dsm_png}'.format(eval_dsm=eval_dsm, eval_dsm_png=eval_dsm + '.png')
            run_cmd(cmd)

        # evaluate for core3d
        # cmd = 'python3 /home/cornell/kz298/core3d-metrics/core3dmetrics/run_geometrics.py --test-ignore 2 \
        #         -c {evaluate_config}'.format(evaluate_config=self.config['evaluate_config'])
        # run_cmd(cmd)

        # compute a difference mask
        # cmd = 'gdal_calc.py -A {evaluate_dir}/eval_dsm_0.5.tif\
        #                     -B {evaluate_dir}/eval_ground_truth_0.5.tif \
        #                     --outfile error.tif --calc="abs(A-B)"'.format(evaluate_dir=evaluate_dir)
        # run_cmd(cmd)

        # evaluate for mvs3dm
        cmd = '/bigdata/kz298/dataset/mvs3dm/Challenge_Data_and_Software/software/masterchallenge_metrics/build/bin/run-metrics \
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
