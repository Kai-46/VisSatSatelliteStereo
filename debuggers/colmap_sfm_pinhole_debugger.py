import os
from rpc_triangulate_solver.triangulate import triangulate
import numpy as np
from colmap.extract_sfm import extract_all_to_dir
from visualization.plot_reproj_err import plot_reproj_err
from debuggers.check_align import check_align
from debuggers.inspect_sfm import SparseInspector
import logging
from coordinate_system import global_to_local


def make_subdirs(sfm_dir):
    subdirs = [
                sfm_dir,
                os.path.join(sfm_dir, 'init_triangulate')
    ]

    for item in subdirs:
        if not os.path.exists(item):
            os.mkdir(item)


def check_sfm(work_dir, sfm_dir, warping_file):
    subdirs = [
                os.path.join(sfm_dir, 'init_triangulate'),
    ]

    for dir in subdirs:
        logging.info('\ninspecting {} ...'.format(dir))
        inspect_dir = dir + '_inspect'
        sfm_inspector = SparseInspector(dir, inspect_dir, camera_model='PINHOLE')
        sfm_inspector.inspect_all()

        # _, xyz_file, track_file = extract_all_to_dir(dir, inspect_dir)
        #
        # # triangulate points
        # meta_file = os.path.join(work_dir, 'metas.json')
        # affine_file = os.path.join(work_dir, 'approx_camera/affine_latlonalt.json')
        # track_file = os.path.join(inspect_dir, 'kai_tracks.json')
        # out_file = os.path.join(inspect_dir, 'kai_latlonalt_coordinates.txt')
        # tmp_dir = os.path.join(inspect_dir, 'tmp')
        # triangulate(meta_file, affine_file, track_file, out_file, tmp_dir)
        #
        # # check rpc reprojection error
        # latlonalterr = np.loadtxt(out_file)
        # plot_reproj_err(latlonalterr[:, 3], os.path.join(inspect_dir, 'rpc_reproj_err.jpg'))
        #
        # # check alignment
        # source = np.loadtxt(xyz_file)[:, 0:3]
        #
        # # convert lat lon alt to local
        # xx, yy, zz = global_to_local(work_dir, latlonalterr[:, 0:1], latlonalterr[:, 1:2], latlonalterr[:, 2:3])
        # target = np.hstack((xx, yy, zz))
        #

        # check_align(source, target)
