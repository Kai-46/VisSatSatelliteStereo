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

import os
import numpy as np
from colmap_sfm_utils import write_template_pinhole
import json
import colmap_sfm_commands
from colmap.extract_sfm import extract_camera_dict, write_all_tracks


def make_subdirs(sfm_dir):
    subdirs = [
                sfm_dir,
                os.path.join(sfm_dir, 'init_triangulate'),
    ]

    for item in subdirs:
        if not os.path.exists(item):
            os.mkdir(item)


def run_sfm(work_dir, sfm_dir, init_camera_file):
    make_subdirs(sfm_dir)

    with open(init_camera_file) as fp:
        init_camera_dict = json.load(fp)
    with open(os.path.join(sfm_dir, 'init_camera_dict.json'), 'w') as fp:
        json.dump(init_camera_dict, fp, indent=2, sort_keys=True)

    init_template = os.path.join(sfm_dir, 'init_template.json')
    write_template_pinhole(init_camera_dict, init_template)

    img_dir = os.path.join(sfm_dir, 'images')
    db_file = os.path.join(sfm_dir, 'database.db')

    colmap_sfm_commands.run_sift_matching(img_dir, db_file, camera_model='PINHOLE')

    out_dir = os.path.join(sfm_dir, 'init_triangulate')
    colmap_sfm_commands.run_point_triangulation(img_dir, db_file, out_dir, init_template, 1.5, 2.0, 2.0)

    # # global bundle adjustment, this might not be useful, just left here for comparison
    # in_dir = os.path.join(sfm_dir, 'init_triangulate')
    # out_dir = os.path.join(sfm_dir, 'init_triangulate_ba')
    # colmap_sfm_commands.run_global_ba(in_dir, out_dir)

    # write final camera_dict
    # camera_dict = extract_camera_dict(out_dir)
    # with open(os.path.join(sfm_dir, 'final_camera_dict.json'), 'w') as fp:
    #     json.dump(camera_dict, fp, indent=2, sort_keys=True)

    # normalize for the MVS
    # colmap_dir = os.path.join(work_dir, 'colmap')
    # in_dir = os.path.join(sfm_dir, 'init_triangulate_ba')
    # out_dir = os.path.join(colmap_dir, 'sparse_for_mvs')
    # tform_file = os.path.join(colmap_dir, 'normalize.txt')
    # if not os.path.exists(out_dir):
    #     os.mkdir(out_dir)
    # colmap_sfm_commands.run_normalize(in_dir, out_dir, tform_file)

    # make a symbolic link here
    # colmap_dir = os.path.join(work_dir, 'colmap')
    # if os.path.exists(os.path.join(colmap_dir, 'sparse_for_mvs')):
    #     os.unlink(os.path.join(colmap_dir, 'sparse_for_mvs'))
    # os.symlink(os.path.join(sfm_dir, 'init_triangulate'), os.path.join(colmap_dir, 'sparse_for_mvs'))
