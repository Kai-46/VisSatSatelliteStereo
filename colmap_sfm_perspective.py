import os
from write_template import write_template_perspective
from absolute_coordinate import triangualte_all_points
import numpy as np
from correct_init import correct_init
import json
import colmap_sfm_commands
from colmap.extract_sfm import extract_camera_dict, extract_all_to_dir

# for inspector
from lib.ply_np_converter import np2ply
from inspector.plot_reproj_err import plot_reproj_err
from check_align import check_align
from inspector.inspect_sfm import SparseInspector
import logging
import utm
import shutil


def make_subdirs(sfm_dir):
    subdirs = [
                sfm_dir,
                os.path.join(sfm_dir, 'init_triangulate'),
                os.path.join(sfm_dir, 'init_triangulate_ba'),
                os.path.join(sfm_dir, 'init_ba_triangulate')
    ]

    for item in subdirs:
        if not os.path.exists(item):
            os.mkdir(item)


def check_sfm(work_dir, sfm_dir):
    subdirs = [
                os.path.join(sfm_dir, 'init_triangulate'),
                os.path.join(sfm_dir, 'init_triangulate_ba'),
                os.path.join(sfm_dir, 'init_ba_triangulate')
    ]

    with open(os.path.join(work_dir, 'aoi.json')) as fp:
        aoi_dict = json.load(fp)
    aoi_ll_east = aoi_dict['ul_easting']
    aoi_ll_north = aoi_dict['ul_northing'] - aoi_dict['height']
    zone_number = aoi_dict['zone_number']
    hemisphere = aoi_dict['hemisphere']
    northern = True if hemisphere == 'N' else False

    for dir in subdirs:
        logging.info('\ninspecting {} ...'.format(dir))
        inspect_dir = dir + '_inspect'

        _, xyz_file, track_file = extract_all_to_dir(dir, inspect_dir)

        xyz_global = np.loadtxt(xyz_file)
        xyz_global[:, 0] += aoi_ll_east
        xyz_global[:, 1] += aoi_ll_north
        np.savetxt(os.path.join(inspect_dir, 'kai_coordinates_utm.txt'), xyz_global,
                   header='# format: easting, northing, height, reproj_err')

        # convert to latitue, longitude
        for kk in range(xyz_global.shape[0]):
            lat, lon = utm.to_latlon(xyz_global[kk, 0], xyz_global[kk, 1], zone_number, northern=northern)
            xyz_global[kk, 0] = lat
            xyz_global[kk, 1] = lon
        np.savetxt(os.path.join(inspect_dir, 'kai_coordinates_latlon.txt'), xyz_global,
                   header='# format: lat, lon, height, reproj_err')

        # triangulate points
        rpc_xyz_file = os.path.join(inspect_dir, 'kai_rpc_coordinates.txt')
        triangualte_all_points(work_dir, track_file, rpc_xyz_file, os.path.join(inspect_dir, '/tmp'))

        sfm_inspector = SparseInspector(dir, inspect_dir, camera_model='PERSPECTIVE')
        sfm_inspector.inspect_all()

        # check rpc_reproj_err and rpc_points
        rpc_coordinates = np.loadtxt(rpc_xyz_file)
        zone_number = rpc_coordinates[0, 3]
        hemisphere = 'N' if rpc_coordinates[0, 4] > 0 else 'S'
        comments = ['projection: UTM {}{}'.format(zone_number, hemisphere),]
        np2ply(rpc_coordinates[:, 0:3], os.path.join(inspect_dir, 'rpc_absolute_points.ply'), comments)
        plot_reproj_err(rpc_coordinates[:, -1], os.path.join(inspect_dir, 'rpc_reproj_err.jpg'))

        # check alignment
        source = np.loadtxt(xyz_file)[:, 0:3]
        target = rpc_coordinates[:, 0:3]
        check_align(work_dir, source, target)

        # write to local
        target = np.concatenate((rpc_coordinates[:, 0:3], rpc_coordinates[:, 5:6]), axis=1)
        target[:, 0] -= aoi_ll_east
        target[:, 1] -= aoi_ll_north

        out_file = os.path.join(inspect_dir, 'kai_rpc_coordinates_local.txt')
        np.savetxt(out_file, target,
                   header='# format: easting, northing, height, reproj_err')


def remove_outliers(sparse_dir, bbx):
    x_min, x_max, y_min, y_max, z_min, z_max = bbx

    os.rename(os.path.join(sparse_dir, 'points3D.txt'), os.path.join(sparse_dir, 'points3D.txt.bak'))
    with open(os.path.join(sparse_dir, 'points3D.txt.bak')) as fp:
        lines = fp.readlines()

    remove_cnt = 0
    remove_ids = []
    original_cnt = len(lines) - 3
    new_lines = lines[0:3]
    for i in range(3, len(lines)):
        tmp = lines[i].split(' ')
        # remove points that lie out of bounding box
        if len(tmp) > 1:
            x = float(tmp[1])
            y = float(tmp[2])
            z = float(tmp[3])

            if x < x_min or x > x_max or y < y_min or y > y_max or z < z_min or z > z_max:
                remove_ids.append(int(tmp[0]))
                remove_cnt += 1
            else:
                new_lines.append(lines[i])

    logging.info('\n\nremoved {}/{} ({:.3} %) outliers that lie out of the bounding box\n\n'.format(remove_cnt, original_cnt,
                                                                                            remove_cnt / original_cnt * 100.0))
    with open(os.path.join(sparse_dir, 'points3D.txt'), 'w') as fp:
        fp.writelines(new_lines)

    os.rename(os.path.join(sparse_dir, 'images.txt'), os.path.join(sparse_dir, 'images.txt.bak'))
    with open(os.path.join(sparse_dir, 'images.txt.bak')) as fp:
        lines = fp.readlines()

    new_lines = lines[0:4]
    for i in range(4, len(lines)):
        tmp = lines[i].split(' ')
        # remove points that lie out of bounding box
        if (i - 4) % 2 == 0:
            new_lines.append(lines[i])
        else:
            kk = 2
            while kk < len(tmp):
                id = int(tmp[kk])
                if id != -1 and id in remove_ids:
                    tmp[kk] = '-1'
                kk += 3
            new_lines.append(' '.join(tmp))

    with open(os.path.join(sparse_dir, 'images.txt'), 'w') as fp:
        fp.writelines(new_lines)


def run_sfm(work_dir, sfm_dir, init_camera_file):
    make_subdirs(sfm_dir)

    with open(init_camera_file) as fp:
        init_camera_dict = json.load(fp)
    with open(os.path.join(sfm_dir, 'init_camera_dict.json'), 'w') as fp:
        json.dump(init_camera_dict, fp, indent=2, sort_keys=True)
    init_template = os.path.join(sfm_dir, 'init_template.json')
    write_template_perspective(init_camera_dict, init_template)

    img_dir = os.path.join(sfm_dir, 'images')
    db_file = os.path.join(sfm_dir, 'database.db')

    colmap_sfm_commands.run_sift_matching(img_dir, db_file, camera_model='PERSPECTIVE')

    out_dir = os.path.join(sfm_dir, 'init_triangulate')
    colmap_sfm_commands.run_point_triangulation(img_dir, db_file, out_dir, init_template, 1.5, 2, 2)

    # we change target to a local coordinate frame
    with open(os.path.join(work_dir, 'aoi.json')) as fp:
        aoi_dict = json.load(fp)
    bbx = (0.0, aoi_dict['width'], 0.0, aoi_dict['height'], 0.0, 100.0)
    remove_outliers(os.path.join(sfm_dir, 'init_triangulate'), bbx)

    # global bundle adjustment
    in_dir = os.path.join(sfm_dir, 'init_triangulate')
    out_dir = os.path.join(sfm_dir, 'init_triangulate_ba')
    colmap_sfm_commands.run_global_ba(in_dir, out_dir)

    remove_outliers(os.path.join(sfm_dir, 'init_triangulate_ba'), bbx)

    # retriangulate
    camera_dict = extract_camera_dict(out_dir)
    with open(os.path.join(sfm_dir, 'init_ba_camera_dict.json'), 'w') as fp:
        json.dump(camera_dict, fp, indent=2, sort_keys=True)
    init_ba_template = os.path.join(sfm_dir, 'init_ba_template.json')
    write_template_perspective(camera_dict, init_ba_template)

    out_dir = os.path.join(sfm_dir, 'init_ba_triangulate')
    colmap_sfm_commands.run_point_triangulation(img_dir, db_file, out_dir, init_ba_template, 1.5, 2, 2)

    remove_outliers(os.path.join(sfm_dir, 'init_ba_triangulate'), bbx)

    # copy
    shutil.copyfile(os.path.join(sfm_dir, 'init_ba_camera_dict.json'), os.path.join(sfm_dir, 'final_camera_dict.json'))
