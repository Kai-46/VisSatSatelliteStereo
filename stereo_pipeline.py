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


import os
import json
from clean_data import clean_data
from image_crop import image_crop
from camera_approx import CameraApprox
import colmap_sfm_perspective
import shutil
import logging
from lib.timer import Timer
from lib.logger import GlobalLogger
from reparam_depth import reparam_depth
from colmap_mvs_commands import run_photometric_mvs, run_consistency_check
import aggregate_2p5d
import aggregate_3d
import utm
from debuggers.inspect_sfm import SparseInspector
import multiprocessing
from datetime import datetime


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

        self.write_aoi()

        per_step_time = []  # (whether to run, step name, time in minutes)

        if self.config['steps_to_run']['clean_data']:
            start_time = datetime.now()
            self.clean_data()
            duration = (datetime.now() - start_time).total_seconds() / 60.0  # minutes
            per_step_time.append((True, 'clean_data', duration))
            print('step clean_data:\tfinished in {} minutes'.format(duration))
        else:
            per_step_time.append((False, 'clean_data', 0.0))
            print('step clean_data:\tskipped')

        if self.config['steps_to_run']['crop_image']:
            start_time = datetime.now()
            self.run_crop_image()
            duration = (datetime.now() - start_time).total_seconds() / 60.0  # minutes
            per_step_time.append((True, 'crop_image', duration))
            print('step crop_image:\tfinished in {} minutes'.format(duration))
        else:
            per_step_time.append((False, 'crop_image', 0.0))
            print('step crop_image:\tskipped')

        if self.config['steps_to_run']['derive_approx']:
            start_time = datetime.now()
            self.run_derive_approx()
            duration = (datetime.now() - start_time).total_seconds() / 60.0  # minutes
            per_step_time.append((True, 'drive_approx', duration))
            print('step derive_approx:\tfinished in {} minutes'.format(duration))
        else:
            per_step_time.append((False, 'drive_approx', 0.0))
            print('step derive_approx:\tskipped')

        if self.config['steps_to_run']['choose_subset']:
            start_time = datetime.now()
            self.run_choose_subset()
            duration = (datetime.now() - start_time).total_seconds() / 60.0  # minutes
            per_step_time.append((True, 'choose_subset', duration))
            print('step choose_subset:\tfinished in {} minutes'.format(duration))
        else:
            per_step_time.append((False, 'choose_subset', 0.0))
            print('step choose_subset:\tskipped')

        if self.config['steps_to_run']['colmap_sfm_perspective']:
            start_time = datetime.now()
            self.run_colmap_sfm_perspective()
            duration = (datetime.now() - start_time).total_seconds() / 60.0  # minutes
            per_step_time.append((True, 'colmap_sfm_perspective', duration))
            print('step colmap_sfm_perspective:\tfinished in {} minutes'.format(duration))
        else:
            per_step_time.append((False, 'colmap_sfm_perspective', 0.0))
            print('step colmap_sfm_perspective:\tskipped')

        if self.config['steps_to_run']['inspect_sfm_perspective']:
            start_time = datetime.now()
            self.run_inspect_sfm_perspective()
            duration = (datetime.now() - start_time).total_seconds() / 60.0  # minutes
            per_step_time.append((True, 'inspect_sfm_perspective', duration))
            print('step inspect_sfm_perspective:\tfinished in {} minutes'.format(duration))
        else:
            per_step_time.append((False, 'inspect_sfm_perspective', 0.0))
            print('step inspect_sfm_perspective:\tskipped')

        if self.config['steps_to_run']['reparam_depth']:
            start_time = datetime.now()
            self.run_reparam_depth()
            duration = (datetime.now() - start_time).total_seconds() / 60.0  # minutes
            per_step_time.append((True, 'reparam_depth', duration))
            print('step reparam_depth:\tfinished in {} minutes'.format(duration))
        else:
            per_step_time.append((False, 'reparam_depth', 0.0))
            print('step reparam_depth:\tskipped')

        if self.config['steps_to_run']['colmap_mvs']:
            start_time = datetime.now()
            self.run_colmap_mvs()
            duration = (datetime.now() - start_time).total_seconds() / 60.0  # minutes
            per_step_time.append((True, 'colmap_mvs', duration))
            print('step colmap_mvs:\tfinished in {} minutes'.format(duration))
        else:
            per_step_time.append((False, 'colmap_mvs', 0.0))
            print('step colmap_mvs:\tskipped')

        if self.config['steps_to_run']['aggregate_2p5d']:
            start_time = datetime.now()
            self.run_aggregate_2p5d()
            duration = (datetime.now() - start_time).total_seconds() / 60.0  # minutes
            per_step_time.append((True, 'aggregate_2p5d', duration))
            print('step aggregate_2p5d:\tfinished in {} minutes'.format(duration))
        else:
            per_step_time.append((False, 'aggregate_2p5d', 0.0))
            print('step aggregate_2p5d:\tskipped')

        if self.config['steps_to_run']['aggregate_3d']:
            start_time = datetime.now()
            self.run_aggregate_3d()
            duration = (datetime.now() - start_time).total_seconds() / 60.0  # minutes
            per_step_time.append((True, 'aggregate_3d', duration))
            print('step aggregate_3d:\tfinished in {} minutes'.format(duration))
        else:
            per_step_time.append((False, 'aggregate_3d', 0.0))
            print('step aggregate_3d:\tskipped')

        with open(os.path.join(self.config['work_dir'], 'runtime.txt'), 'w') as fp:
            fp.write('step_name, status, duration (minutes)\n')
            total = 0.0
            for (has_run, step_name, duration) in per_step_time:
                if has_run:
                    fp.write('{}, success, {}\n'.format(step_name, duration))
                else:
                    fp.write('{}, skipped\n'.format(step_name))
                total += duration
            fp.write('\ntotal: {} minutes\n'.format(total))

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
        image_template = None
        if 'image_template' in self.config:
            image_template = self.config['image_template']

        # set log file
        log_file = os.path.join(work_dir, 'logs/log_crop_image.txt')
        self.logger.set_log_file(log_file)

        # create a local timer
        local_timer = Timer('Image cropping module')
        local_timer.start()

        # crop image and tone map
        image_crop(work_dir)

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
            # create symbolic link to avoid data copy
            os.symlink(os.path.relpath(os.path.join(work_dir, 'images', img_name), image_subdir),
                       os.path.join(image_subdir, img_name))

        with open(os.path.join(out_dir, 'perspective_dict.json'), 'w') as fp:
            json.dump(subset_perspective_dict, fp, indent=2)

    def run_colmap_sfm_perspective(self, weight=0.01):
        work_dir = os.path.abspath(self.config['work_dir'])
        sfm_dir = os.path.join(work_dir, 'colmap/sfm_perspective')
        if not os.path.exists(sfm_dir):
            os.mkdir(sfm_dir)

        log_file = os.path.join(work_dir, 'logs/log_sfm_perspective.txt')
        self.logger.set_log_file(log_file)
        # create a local timer
        local_timer = Timer('Colmap SfM Module, perspective camera')
        local_timer.start()

        # create a hard link to avoid copying of images
        if os.path.exists(os.path.join(sfm_dir, 'images')):
            os.unlink(os.path.join(sfm_dir, 'images'))
        os.symlink(os.path.relpath(os.path.join(work_dir, 'colmap/subset_for_sfm/images'), sfm_dir),
                   os.path.join(sfm_dir, 'images'))
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

        for subdir in ['tri', 'tri_ba']:
            dir = os.path.join(sfm_dir, subdir)
            logging.info('\ninspecting {} ...'.format(dir))

            inspect_dir = os.path.join(sfm_dir, 'inspect_' + subdir)
            if os.path.exists(inspect_dir):
                shutil.rmtree(inspect_dir)

            db_path = os.path.join(sfm_dir, 'database.db')
            sfm_inspector = SparseInspector(dir, db_path, inspect_dir, camera_model='PERSPECTIVE')
            sfm_inspector.inspect_all()

        # stop local timer
        local_timer.mark('inspect sfm perspective done')
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

        # link to sfm_perspective
        if os.path.exists(os.path.join(mvs_dir, 'images')):
           os.unlink(os.path.join(mvs_dir, 'images'))
        os.symlink(os.path.relpath(os.path.join(colmap_dir, 'sfm_perspective/images'), mvs_dir),
                  os.path.join(mvs_dir, 'images'))

        if os.path.exists(os.path.join(mvs_dir, 'sparse')):
           os.unlink(os.path.join(mvs_dir, 'sparse'))
        os.symlink(os.path.relpath(os.path.join(colmap_dir, 'sfm_perspective/tri_ba'), mvs_dir),
                   os.path.join(mvs_dir, 'sparse'))

        # compute depth ranges and generate last_rows.txt
        reparam_depth(os.path.join(mvs_dir, 'sparse'), mvs_dir, camera_model='perspective')

        # prepare stereo directory
        stereo_dir = os.path.join(mvs_dir, 'stereo')
        for subdir in [stereo_dir, 
                       os.path.join(stereo_dir, 'depth_maps'),
                       os.path.join(stereo_dir, 'normal_maps'),
                       os.path.join(stereo_dir, 'consistency_graphs')]:
            if not os.path.exists(subdir):
                os.mkdir(subdir)

        # write patch-match.cfg and fusion.cfg
        image_names = sorted(os.listdir(os.path.join(mvs_dir, 'images')))

        with open(os.path.join(stereo_dir, 'patch-match.cfg'), 'w') as fp:
            for img_name in image_names:
                fp.write(img_name + '\n__auto__, 20\n')
                
                # use all images
                # fp.write(img_name + '\n__all__\n')

                # randomly choose 20 images
                # from random import shuffle
                # candi_src_images = [x for x in image_names if x != img_name]
                # shuffle(candi_src_images)
                # max_src_images = 10
                # fp.write(img_name + '\n' + ', '.join(candi_src_images[:max_src_images]) + '\n')

        with open(os.path.join(stereo_dir, 'fusion.cfg'), 'w') as fp:
            for img_name in image_names:
                fp.write(img_name + '\n')

        # stop local timer
        local_timer.mark('reparam depth done')
        logging.info(local_timer.summary())

    def run_colmap_mvs(self, window_radius=3):
        work_dir = self.config['work_dir']
        mvs_dir = os.path.join(work_dir, 'colmap/mvs')

        # set log file
        log_file = os.path.join(work_dir, 'logs/log_mvs.txt')
        self.logger.set_log_file(log_file)
        # create a local timer
        local_timer = Timer('Colmap MVS Module')
        local_timer.start()

        # first run PMVS without filtering
        run_photometric_mvs(mvs_dir, window_radius)

        # next do forward-backward checking and filtering
        run_consistency_check(mvs_dir, window_radius)

        # stop local timer
        local_timer.mark('Colmap MVS done')
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

        max_processes = -1
        if 'aggregate_max_processes' in self.config:
            max_processes = self.config['aggregate_max_processes']

        aggregate_2p5d.run_fuse(work_dir, max_processes=max_processes)

        # stop local timer
        local_timer.mark('2.5D aggregation done')
        logging.info(local_timer.summary())


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Satellite Stereo')
    parser.add_argument('--config_file', type=str,
                        help='configuration file')

    args = parser.parse_args()

    pipeline = StereoPipeline(args.config_file)

    pipeline.run()
