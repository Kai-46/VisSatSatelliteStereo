# examples/Python/Basic/icp_registration.py

from open3d import *
import numpy as np
import copy
from lib.dsm_np_converter import dsm2np
from lib.ply_np_converter import np2ply
from lib.dsm_util import read_dsm_tif, write_dsm_tif
from lib.proj_to_grid import proj_to_grid


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    draw_geometries([source_temp, target_temp])


def align_dsm_tif(dsm_source, dsm_target):
    source = dsm2np(dsm_source)
    target = dsm2np(dsm_target)


if __name__ == "__main__":

    dsm_source = '/data2/kz298/mvs3dm_result/MasterProvisional2/mvs_results_all/aggregate_2p5d/evaluation/aggregate_2p5d.tif'
    dsm_target = '/data2/kz298/mvs3dm_result/MasterProvisional2/mvs_results_all/aggregate_2p5d/evaluation/eval_ground_truth.tif'
    source_pts = dsm2np(dsm_source)
    target_pts = dsm2np(dsm_target)

    source_file = dsm_source[:-4] + '.ply'
    target_file = dsm_target[:-4] + '.ply'

    global_x_shift = np.min(target_pts[:, 0]) - 20
    global_y_shift = np.min(target_pts[:, 1]) - 20

    source_pts[:, 0] -= global_x_shift
    source_pts[:, 1] -= global_y_shift

    target_pts[:, 0] -= global_x_shift
    target_pts[:, 1] -= global_y_shift

    np2ply(source_pts, source_file)
    np2ply(target_pts, target_file)

    source = read_point_cloud(source_file)
    target = read_point_cloud(target_file)

    threshold = 0.4
    trans_init = np.asarray(
                [[1.0, 0.0, 0.0,  0.0],
                [0.0, 1.0, 0.0,  0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0]])
    draw_registration_result(source, target, trans_init)
    print("Initial alignment")
    evaluation = evaluate_registration(source, target,
            threshold, trans_init)
    print(evaluation)

    print("Apply point-to-point ICP")
    reg_p2p = registration_icp(source, target, threshold, trans_init,
            TransformationEstimationPointToPoint())
    print(reg_p2p)
    print("Transformation is:")
    print(reg_p2p.transformation)
    print("")
    draw_registration_result(source, target, reg_p2p.transformation)

    # print("Apply point-to-plane ICP")
    # reg_p2l = registration_icp(source, target, threshold, trans_init,
    #         TransformationEstimationPointToPlane())
    # print(reg_p2l)
    # print("Transformation is:")
    # print(reg_p2l.transformation)
    # print("")
    # draw_registration_result(source, target, reg_p2l.transformation)

    # apply transformation
    source.transform(reg_p2p.transformation)
    xyz = np.asarray(source.points)
    np2ply(source_pts, source_file[:-4] + '_icp.ply')

    xyz[:, 0] += global_x_shift
    xyz[:, 1] += global_y_shift

    _, meta_dict = read_dsm_tif(dsm_target)

    dsm = proj_to_grid(xyz, meta_dict['ul_easting'], meta_dict['ul_northing'], meta_dict['east_resolution'], meta_dict['north_resolution'],
                       meta_dict['img_width'], meta_dict['img_height'])

    write_dsm_tif(dsm, meta_dict, dsm_source[:-4]+'_icp.tif')
