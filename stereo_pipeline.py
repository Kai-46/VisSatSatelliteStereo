import multiprocessing
import os
import json
from clean_data import clean_data
from image_crop import image_crop
from camera_approx import CameraApprox
import colmap_sfm_perspective, colmap_sfm_pinhole
from skew_correct import skew_correct
import shutil
import logging
from lib.run_cmd import run_cmd
from lib.timer import Timer
import numpy as np
from lib.logger import GlobalLogger
from reparam_depth import reparam_depth
from colmap_mvs_commands import run_photometric_mvs, run_consistency_check
from process_dsm_gt import process_dsm_gt
import aggregate_2p5d
import aggregate_3d
import utm


class StereoPipeline(object):
    def __init__(self, config_file):
        with open(config_file) as fp:
            self.config = json.load(fp)

        self.pan_msi_pairing = None

        self.crop_image_max_processes = multiprocessing.cpu_count()
        if 'crop_image_max_processes' in self.config:
            self.crop_image_max_processes = self.config['crop_image_max_processes']

        # make work_dir
        if not os.path.exists(self.config['work_dir']):
            os.mkdir(self.config['work_dir'])
        logs_subdir = os.path.join(self.config['work_dir'], 'logs')
        if not os.path.exists(logs_subdir):
            os.mkdir(logs_subdir)

        self.logger = GlobalLogger()

    def run(self):
        print(self.config)

        self.write_aoi()

        if 'pan_msi_pairing' in self.config:
            self.pan_msi_pairing = self.config['pan_msi_pairing']

        if self.config['steps_to_run']['clean_data']:
            self.clean_data()

        if self.config['steps_to_run']['process_dsm_gt']:
            process_dsm_gt(self.config['work_dir'], self.config['ground_truth'])

        if self.config['steps_to_run']['crop_image']:
            self.run_crop_image()

        if self.config['steps_to_run']['derive_approx']:
            self.run_derive_approx()

        if self.config['steps_to_run']['choose_subset']:
            self.run_choose_subset()

        if self.config['steps_to_run']['colmap_sfm_perspective']:
            self.run_colmap_sfm_perspective()

        if self.config['steps_to_run']['inspect_sfm_perspective']:
            self.run_inspect_sfm_perspective()

        if self.config['steps_to_run']['debug_approx']:
            self.run_debug_approx()

        if self.config['steps_to_run']['skew_correct']:
            self.run_skew_correct()

        if self.config['steps_to_run']['select_subset']:
            self.run_select_subset()

        if self.config['steps_to_run']['colmap_sfm_pinhole']:
            self.run_colmap_sfm_pinhole()

        if self.config['steps_to_run']['inspect_sfm_pinhole']:
            self.run_inspect_sfm_pinhole()

        if self.config['steps_to_run']['reparam_depth']:
            self.run_reparam_depth()

        if self.config['steps_to_run']['colmap_mvs']:
            self.run_colmap_mvs()

        if self.config['steps_to_run']['inspect_mvs']:
            self.run_inspect_mvs()

        if self.config['steps_to_run']['aggregate_2p5d']:
            self.run_aggregate_2p5d()

        if self.config['steps_to_run']['aggregate_3d']:
            self.run_aggregate_3d()

        if self.config['steps_to_run']['evaluate']:
            self.run_evaluation()

    def write_aoi(self):
        # write aoi.json
        bbx_utm = self.config['bounding_box']
        zone_number = bbx_utm['zone_number']
        hemisphere = bbx_utm['hemisphere']
        ul_easting = bbx_utm['ul_easting']
        ul_northing = bbx_utm['ul_northing']
        lr_easting = ul_easting + bbx_utm['width']
        lr_northing = ul_northing - bbx_utm['height']

        # compute a lat_lon bbx
        corners_easting = [ul_easting, lr_easting, lr_easting, ul_easting]
        corners_northing = [ul_northing, ul_northing, lr_northing, lr_northing]
        corners_lat = []
        corners_lon = []
        northern = True if hemisphere == 'N' else False
        for i in range(4):
            lat, lon = utm.to_latlon(corners_easting[i], corners_northing[i], zone_number, northern=northern)
            corners_lat.append(lat)
            corners_lon.append(lon)
        lat_min = min(corners_lat)
        lat_max = max(corners_lat)
        lon_min = min(corners_lon)
        lon_max = max(corners_lon)

        aoi_dict = {'zone_number': zone_number,
                    'hemisphere': hemisphere,
                    'ul_easting': ul_easting,
                    'ul_northing': ul_northing,
                    'lr_easting': lr_easting,
                    'lr_northing': lr_northing,
                    'width': bbx_utm['width'],
                    'height': bbx_utm['height'],
                    'lat_min': lat_min,
                    'lat_max': lat_max,
                    'lon_min': lon_min,
                    'lon_max': lon_max,
                    'alt_min': self.config['alt_min'],
                    'alt_max': self.config['alt_max']}

        with open(os.path.join(self.config['work_dir'], 'aoi.json'), 'w') as fp:
            json.dump(aoi_dict, fp, indent=2)

    def clean_data(self):
        dataset_dir = []
        if self.pan_msi_pairing is not None:
            # if pan_msi_pairing present - build the dataset_dir from the pan files
            for f in self.pan_msi_pairing:
                d = os.path.dirname(f[0])
                if d not in dataset_dir:
                    dataset_dir.append(d)
        elif 'dataset_dir' in self.config:
            dataset_dir = self.config['dataset_dir']

        work_dir = self.config['work_dir']

        # set log file and timer
        log_file = os.path.join(work_dir, 'logs/log_clean_data.txt')
        self.logger.set_log_file(log_file)
        # create a local timer
        local_timer = Timer('Data cleaning Module')
        local_timer.start()

        # clean data
        cleaned_data_dir = os.path.join(work_dir, 'cleaned_data')
        if os.path.exists(cleaned_data_dir):  # remove cleaned_data_dir
            shutil.rmtree(cleaned_data_dir)
        os.mkdir(cleaned_data_dir)

        # check if dataset_dir is a list or tuple
        if not (isinstance(dataset_dir, list) or isinstance(dataset_dir, tuple)):
            dataset_dir = [dataset_dir, ]
        clean_data(dataset_dir, cleaned_data_dir)

        # stop local timer
        local_timer.mark('Data cleaning done')
        logging.info(local_timer.summary())

    def run_crop_image(self):
        work_dir = self.config['work_dir']

        # set log file
        log_file = os.path.join(work_dir, 'logs/log_crop_image.txt')
        self.logger.set_log_file(log_file)

        # create a local timer
        local_timer = Timer('Image cropping module')
        local_timer.start()

        # crop image and tone map
        image_crop(work_dir, self.crop_image_max_processes, self.pan_msi_pairing)

        # stop local timer
        local_timer.mark('image cropping done')
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
        appr = CameraApprox(work_dir)

        appr.approx_perspective_utmalt()
        appr.approx_affine_latlonalt()
        appr.approx_perspective_enu()

        # stop local timer
        local_timer.mark('Derive approximation done')
        logging.info(local_timer.summary())

    def run_choose_subset(self):
        work_dir = os.path.abspath(self.config['work_dir'])
        colmap_dir = os.path.join(work_dir, 'colmap')
        if not os.path.exists(colmap_dir):
            os.mkdir(colmap_dir)
        out_dir = os.path.join(colmap_dir, 'subset_for_sfm')
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)
        os.mkdir(out_dir)
        image_subdir = os.path.join(out_dir, 'images')
        os.mkdir(image_subdir)

        with open(os.path.join(work_dir, 'approx_camera/perspective_enu.json')) as fp:
            perspective_dict = json.load(fp)

        # build image id to name mapping
        img_id2name = {}
        for img_name in perspective_dict.keys():
            id = int(img_name[:img_name.find('_')])
            img_id2name[id] = img_name

        #subset_img_ids = list(range(15)) + list(range(33, 40))
        subset_img_ids = img_id2name.keys()     # select all
        subset_perspective_dict = {}

        for img_id in subset_img_ids:
            img_name = img_id2name[img_id]
            subset_perspective_dict[img_name] = perspective_dict[img_name]
            shutil.copy2(os.path.join(work_dir, 'images', img_name),
                         image_subdir)

        with open(os.path.join(out_dir, 'perspective_dict.json'), 'w') as fp:
            json.dump(subset_perspective_dict, fp, indent=2)

    def run_colmap_sfm_perspective(self, weight=1):
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
        os.symlink(os.path.join(work_dir, 'colmap/subset_for_sfm/images'), os.path.join(colmap_dir, 'sfm_perspective/images'))
        init_camera_file = os.path.join(work_dir, 'colmap/subset_for_sfm/perspective_dict.json')
        colmap_sfm_perspective.run_sfm(work_dir, sfm_dir, init_camera_file, weight)

        # stop local timer
        local_timer.mark('Colmap SfM done')
        logging.info(local_timer.summary())

    def run_inspect_sfm_perspective(self):
        work_dir = os.path.abspath(self.config['work_dir'])

        log_file = os.path.join(work_dir, 'logs/log_inspect_sfm_perspective.txt')
        self.logger.set_log_file(log_file)
        local_timer = Timer('inspect sfm')
        local_timer.start()

        # inspect sfm perspective
        sfm_dir = os.path.join(work_dir, 'colmap/sfm_perspective')
        import debuggers.colmap_sfm_perspective_debugger
        debuggers.colmap_sfm_perspective_debugger.check_sfm(work_dir, sfm_dir)

        # stop local timer
        local_timer.mark('inspect sfm perspective done')
        logging.info(local_timer.summary())

    def run_debug_approx(self):
        work_dir = os.path.abspath(self.config['work_dir'])

        log_file = os.path.join(work_dir, 'logs/log_debug_approx.txt')
        self.logger.set_log_file(log_file)
        local_timer = Timer('debug approx')
        local_timer.start()

        import debuggers.perspective_approx_debugger
        debuggers.perspective_approx_debugger.debug_approx(work_dir)

        # stop local timer
        local_timer.mark('debug approx done')
        logging.info(local_timer.summary())

    def run_skew_correct(self):
        work_dir = os.path.abspath(self.config['work_dir'])
        colmap_dir = os.path.join(work_dir, 'colmap')
        subdirs = [
            colmap_dir,
            os.path.join(colmap_dir, 'skew_correct')
        ]

        for item in subdirs:
            if not os.path.exists(item):
                os.mkdir(item)

        # second time SfM
        log_file = os.path.join(work_dir, 'logs/log_skew_correct.txt')
        self.logger.set_log_file(log_file)
        # create a local timer
        local_timer = Timer('Skew correct module')
        local_timer.start()

        # skew-correct images
        skew_correct(work_dir)

        # stop local timer
        local_timer.mark('Skew correct done')
        logging.info(local_timer.summary())

    def run_select_subset(self):
        work_dir = os.path.abspath(self.config['work_dir'])
        out_dir = os.path.join(work_dir, 'colmap/subset_for_mvs')
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)
        os.mkdir(out_dir)

        image_subdir = os.path.join(out_dir, 'images')
        if not os.path.exists(image_subdir):
            os.mkdir(image_subdir)

        log_file = os.path.join(work_dir, 'logs/log_select_subset.txt')
        self.logger.set_log_file(log_file)
        # create a local timer
        local_timer = Timer('select subset')
        local_timer.start()

        with open(os.path.join(work_dir, 'colmap/skew_correct/pinhole_dict.json')) as fp:
            pinhole_dict = json.load(fp)

        # build image id to name mapping
        img_id2name = {}
        for img_name in pinhole_dict.keys():
            id = int(img_name[:img_name.find('_')])
            img_id2name[id] = img_name

        #subset_img_ids = list(range(15)) + list(range(33, 40))
        subset_img_ids = img_id2name.keys()     # select all
        subset_pinhole_dict = {}

        for img_id in subset_img_ids:
            img_name = img_id2name[img_id]
            subset_pinhole_dict[img_name] = pinhole_dict[img_name]
            shutil.copy2(os.path.join(work_dir, 'colmap/skew_correct/images', img_name),
                         image_subdir)

        with open(os.path.join(out_dir, 'pinhole_dict.json'), 'w') as fp:
            json.dump(subset_pinhole_dict, fp, indent=2)

        # stop local timer
        local_timer.mark('select_subset')
        logging.info(local_timer.summary())

    def run_colmap_sfm_pinhole(self):
        work_dir = os.path.abspath(self.config['work_dir'])

        log_file = os.path.join(work_dir, 'logs/log_sfm_pinhole.txt')
        self.logger.set_log_file(log_file)
        # create a local timer
        local_timer = Timer('Colmap SfM Module, pinhole camera')
        local_timer.start()

        # create a hard link to avoid copying of images
        colmap_dir = os.path.join(work_dir, 'colmap')
        sfm_dir = os.path.join(colmap_dir, 'sfm_pinhole')
        if not os.path.exists(sfm_dir):
            os.mkdir(sfm_dir)
        if os.path.exists(os.path.join(colmap_dir, 'sfm_pinhole/images')):
            os.unlink(os.path.join(colmap_dir, 'sfm_pinhole/images'))
        os.symlink(os.path.join(colmap_dir, 'subset_for_mvs/images'), os.path.join(colmap_dir, 'sfm_pinhole/images'))
        init_camera_file = os.path.join(colmap_dir, 'subset_for_mvs/pinhole_dict.json')

        colmap_sfm_pinhole.run_sfm(work_dir, sfm_dir, init_camera_file)

        # stop local timer
        local_timer.mark('Colmap SfM done')
        logging.info(local_timer.summary())

    def run_inspect_sfm_pinhole(self):
        work_dir = os.path.abspath(self.config['work_dir'])
        log_file = os.path.join(work_dir, 'logs/log_inspect_sfm_pinhole.txt')
        self.logger.set_log_file(log_file)
        local_timer = Timer('inspect sfm')
        local_timer.start()

        sfm_dir = os.path.join(work_dir, 'colmap/sfm_pinhole')
        warping_file = os.path.join(work_dir, 'colmap/skew_correct/affine_warpings.json')
        import debuggers.colmap_sfm_pinhole_debugger
        debuggers.colmap_sfm_pinhole_debugger.check_sfm(work_dir, sfm_dir, warping_file)

        # stop local timer
        local_timer.mark('inspect sfm pinhole done')
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
                            --image_path {colmap_dir}/sfm_pinhole/images  \
                            --input_path {colmap_dir}/sfm_pinhole/init_triangulate \
                            --output_path {colmap_dir}/mvs'.format(colmap_dir=colmap_dir)
        run_cmd(cmd)

        # compute depth ranges and generate last_rows.txt
        reparam_depth(os.path.join(mvs_dir, 'sparse'), mvs_dir)

        # stop local timer
        local_timer.mark('reparam depth done')
        logging.info(local_timer.summary())

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
        from convert_mvs_results import convert_depth_maps, convert_normal_maps
        type_name = 'geometric'
        #type_name = 'photometric'
        convert_depth_maps(work_dir, depth_type=type_name)
        convert_normal_maps(work_dir, normal_type=type_name)

        local_timer.mark('inspect mvs done')
        logging.info(local_timer.summary())

    def run_aggregate_3d(self):
        work_dir = self.config['work_dir']
        # set log file
        log_file = os.path.join(work_dir, 'logs/log_aggregate_3d.txt')
        self.logger.set_log_file(log_file)
        # create a local timer
        local_timer = Timer('3D aggregation module')
        local_timer.start()

        aggregate_3d.run_fuse(work_dir)

        # stop local timer
        local_timer.mark('3D aggregation done')
        logging.info(local_timer.summary())

    def run_aggregate_2p5d(self):
        work_dir = self.config['work_dir']
        # set log file
        log_file = os.path.join(work_dir, 'logs/log_aggregate_2p5d.txt')
        self.logger.set_log_file(log_file)
        # create a local timer
        local_timer = Timer('2.5D aggregation module')
        local_timer.start()

        aggregate_2p5d.run_fuse(work_dir)

        # stop local timer
        local_timer.mark('2.5D aggregation done')
        logging.info(local_timer.summary())

    def run_evaluation(self):
        pass


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Satellite Stereo')
    parser.add_argument('--config_file', type=str,
                        help='configuration file')

    args = parser.parse_args()

    pipeline = StereoPipeline(args.config_file)

    pipeline.run()
