import os
from colmap_sfm_utils import write_template_perspective
import json
import colmap_sfm_commands
from colmap.extract_sfm import extract_camera_dict
import logging
import shutil
from debuggers.inspect_sfm import SparseInspector


def make_subdirs(sfm_dir):
    subdirs = [
                sfm_dir,
                os.path.join(sfm_dir, 'tri'),
                os.path.join(sfm_dir, 'tri_ba')
    ]

    for item in subdirs:
        if not os.path.exists(item):
            os.mkdir(item)


def run_sfm(work_dir, sfm_dir, init_camera_file, weight):
    make_subdirs(sfm_dir)

    img_dir = os.path.join(sfm_dir, 'images')
    db_file = os.path.join(sfm_dir, 'database.db')

    colmap_sfm_commands.run_sift_matching(img_dir, db_file, camera_model='PERSPECTIVE')

    with open(init_camera_file) as fp:
        init_camera_dict = json.load(fp)
    with open(os.path.join(sfm_dir, 'init_camera_dict.json'), 'w') as fp:
        json.dump(init_camera_dict, fp, indent=2, sort_keys=True)

    # iterate between triangulation and bundle adjustment
    for reproj_err_threshold in [32.0, 8.0, 2.0]:
        # triangulate
        init_template = os.path.join(sfm_dir, 'init_template.json')
        write_template_perspective(init_camera_dict, init_template)
        tri_dir = os.path.join(sfm_dir, 'tri')
        colmap_sfm_commands.run_point_triangulation(img_dir, db_file, tri_dir, init_template,
                                                    reproj_err_threshold, reproj_err_threshold, reproj_err_threshold)

        # global bundle adjustment
        tri_ba_dir = os.path.join(sfm_dir, 'tri_ba')
        colmap_sfm_commands.run_global_ba(tri_dir, tri_ba_dir, weight)

        # # output statistics
        # inspect_dir = os.path.join(sfm_dir, 'inspect_{}'.format(i))
        # if not os.path.exists(inspect_dir):
        #     os.mkdir(inspect_dir)

        # sfm_inspector = SparseInspector(tri_dir, os.path.join(inspect_dir, 'tri'), camera_model='PERSPECTIVE')
        # sfm_inspector.inspect_all()

        # sfm_inspector = SparseInspector(tri_ba_dir, os.path.join(inspect_dir, 'tri_ba'), camera_model='PERSPECTIVE')
        # sfm_inspector.inspect_all()

        # update camera dict
        init_camera_dict = extract_camera_dict(tri_ba_dir)

    with open(os.path.join(sfm_dir, 'init_ba_camera_dict.json'), 'w') as fp:
        json.dump(init_camera_dict, fp, indent=2, sort_keys=True)

    # for later uses: check how big the image-space translations are
    with open(os.path.join(sfm_dir, 'init_camera_dict.json')) as fp:
        pre_bundle_cameras = json.load(fp)

    with open(os.path.join(sfm_dir, 'init_ba_camera_dict.json')) as fp:
        after_bundle_cameras = json.load(fp)

    result = ['img_name, delta_cx, delta_cy\n', ]
    for img_name in sorted(pre_bundle_cameras.keys()):
        # w, h, fx, fy, cx, cy, s, qw, qx, qy, qz, tx, ty, tz
        pre_bundle_params = pre_bundle_cameras[img_name]
        after_bundle_params = after_bundle_cameras[img_name]
        delta_cx = after_bundle_params[4] - pre_bundle_params[4]
        delta_cy = after_bundle_params[5] - pre_bundle_params[5]

        result.append('{}, {}, {}\n'.format(img_name, delta_cx, delta_cy))

    with open(os.path.join(sfm_dir, 'principal_points_adjustment.csv'), 'w') as fp:
        fp.write(''.join(result))
