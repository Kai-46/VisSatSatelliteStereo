import os
import numpy as np
from write_template import write_template_pinhole
import json
import colmap_sfm_commands
from colmap.extract_sfm import extract_camera_dict, write_all_tracks
from correct_skew import add_skew_to_pinhole_tracks
from absolute_coordinate import triangualte_all_points

# for inspector
from lib.ply_np_converter import np2ply
from inspector.plot_reproj_err import plot_reproj_err
from check_align import check_align
from inspector.inspect_sfm import SparseInspector
import logging


def make_subdirs(sfm_dir):
    subdirs = [
                sfm_dir,
                os.path.join(sfm_dir, 'init_triangulate'),
                os.path.join(sfm_dir, 'init_triangulate_ba')
    ]

    for item in subdirs:
        if not os.path.exists(item):
            os.mkdir(item)


def check_sfm(work_dir, sfm_dir, warping_file):
    subdirs = [
                os.path.join(sfm_dir, 'init_triangulate'),
                os.path.join(sfm_dir, 'init_triangulate_ba')
    ]

    for dir in subdirs:
        logging.info('\ninspecting {} ...'.format(dir))
        inspect_dir = dir + '_inspect'
        if not os.path.exists(inspect_dir):
            os.mkdir(inspect_dir)

        all_tracks = add_skew_to_pinhole_tracks(dir, warping_file)
        xyz_file = os.path.join(inspect_dir, 'kai_coordinates.txt')
        track_file = os.path.join(inspect_dir, 'kai_tracks.json')
        write_all_tracks(all_tracks, xyz_file, track_file)

        # triangulate points
        rpc_xyz_file = os.path.join(inspect_dir, 'kai_rpc_coordinates.txt')
        triangualte_all_points(work_dir, track_file, rpc_xyz_file, os.path.join(inspect_dir, '/tmp'))

        sfm_inspector = SparseInspector(dir, inspect_dir, camera_model='PINHOLE')
        sfm_inspector.inspect_all()

        # check rpc_reproj_err and rpc_points
        rpc_coordinates = np.loadtxt(rpc_xyz_file)
        zone_number = rpc_coordinates[0, 3]
        zone_letter = 'N' if rpc_coordinates[0, 4] > 0 else 'S'
        comments = ['projection: UTM {}{}'.format(zone_number, zone_letter),]
        np2ply(rpc_coordinates[:, 0:3], os.path.join(inspect_dir, 'rpc_absolute_points.ply'), comments)
        plot_reproj_err(rpc_coordinates[:, -1], os.path.join(inspect_dir, 'rpc_reproj_err.jpg'))

        # check alignment
        source = np.loadtxt(xyz_file)[:, 0:3]
        target = rpc_coordinates[:, 0:3]
        check_align(work_dir, source, target)


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
    colmap_sfm_commands.run_point_triangulation(img_dir, db_file, out_dir, init_template)

    # global bundle adjustment, this might not be useful, just left here for comparison
    in_dir = os.path.join(sfm_dir, 'init_triangulate')
    out_dir = os.path.join(sfm_dir, 'init_triangulate_ba')
    colmap_sfm_commands.run_global_ba(in_dir, out_dir)

    # write final camera_dict
    camera_dict = extract_camera_dict(out_dir)
    with open(os.path.join(sfm_dir, 'final_camera_dict.json'), 'w') as fp:
        json.dump(camera_dict, fp, indent=2, sort_keys=True)

    # normalize for the MVS
    colmap_dir = os.path.join(work_dir, 'colmap')
    in_dir = os.path.join(sfm_dir, 'init_triangulate_ba')
    out_dir = os.path.join(colmap_dir, 'sparse_for_mvs')
    tform_file = os.path.join(colmap_dir, 'normalize.txt')
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    colmap_sfm_commands.run_normalize(in_dir, out_dir, tform_file)
