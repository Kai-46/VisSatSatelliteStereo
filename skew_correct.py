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

import multiprocessing

import numpy as np
import json
import logging
import os
import imageio
from lib.warp_affine import warp_affine
from colmap.extract_sfm import extract_all_tracks
import shutil


def skew_correct_worker(work_dir, params, img_name):
    perspective_img_dir = os.path.join(work_dir, 'colmap/sfm_perspective/images')
    out_dir = os.path.join(work_dir, 'colmap/skew_correct')
    pinhole_img_dir = os.path.join(out_dir, 'images')

    fx = params[2]
    fy = params[3]
    cx = params[4]
    cy = params[5]
    s = params[6]
    qvec = params[7:11]
    tvec = params[11:14]

    # compute homography and update s, cx
    norm_skew = s / fy
    cx = cx - s * cy / fy
    # s = 0.

    # warp image
    affine_matrix = np.array([[1., -norm_skew, 0.],
                              [0., 1., 0.]])
    img_src = imageio.imread(os.path.join(perspective_img_dir, img_name))
    img_dst, off_set, affine_matrix = warp_affine(img_src, affine_matrix)
    imageio.imwrite(os.path.join(pinhole_img_dir, img_name), img_dst)

    new_h, new_w = img_dst.shape[:2]
    # add off_set to camera parameters
    cx += off_set[0]
    cy += off_set[1]

    return norm_skew, affine_matrix, [new_w, new_h, fx, fy, cx, cy,
            qvec[0], qvec[1], qvec[2], qvec[3],
            tvec[0], tvec[1], tvec[2]]


def skew_correct(work_dir):
    out_dir = os.path.join(work_dir, 'colmap/skew_correct')
    perspective_img_dir = os.path.join(work_dir, 'colmap/sfm_perspective/images')
    perspective_file = os.path.join(work_dir, 'colmap/sfm_perspective/init_ba_camera_dict.json')

    pinhole_img_dir = os.path.join(out_dir, 'images')
    pinhole_file = os.path.join(out_dir, 'pinhole_dict.json')
    warping_file = os.path.join(out_dir, 'affine_warpings.json')

    if os.path.exists(pinhole_img_dir):
        shutil.rmtree(pinhole_img_dir)
    os.mkdir(pinhole_img_dir)

    with open(perspective_file) as fp:
        perspective_dict = json.load(fp)

    pinhole_dict = {}
    affine_warping_dict = {}
    info_txt = 'img_name, skew (s/fy)\n'

    results = []
    arguments = []
    perspective_images = os.listdir(perspective_img_dir)

    pool_size = min(multiprocessing.cpu_count(), len(perspective_images))
    pool = multiprocessing.Pool(pool_size)

    for img_name in sorted(perspective_images):
        # w, h, fx, fy, cx, cy, s, qvec, t
        params = perspective_dict[img_name]
        w = params[0]
        h = params[1]
        arguments.append((img_name, w, h))
        results.append(pool.apply_async(skew_correct_worker, (work_dir, params, img_name)))

    for i, r in enumerate(results):
        img_name, w, h = arguments[i]
        norm_skew, affine_matrix, pinhole = r.get()
        affine_warping_dict[img_name] = affine_matrix
        logging.info(
            'removed normalized skew: {} in image: {}, original size: {}, {}, new image size: {}, {}'.format(
                norm_skew, img_name, w, h, pinhole[0], pinhole[1]))
        info_txt += '{}, {}\n'.format(img_name, norm_skew)
        pinhole_dict[img_name] = pinhole

    pool.close()
    pool.join()

    with open(pinhole_file, 'w') as fp:
        json.dump(pinhole_dict, fp, indent=2)

    with open(warping_file, 'w') as fp:
        for img_name in sorted(affine_warping_dict.keys()):
            matrix = affine_warping_dict[img_name]
            # fp.write('{} {} {} {} {} {} {}\n'.format(img_name, matrix[0, 0], matrix[0, 1], matrix[0, 2],
            #                                          matrix[1, 0], matrix[1, 1], matrix[1,2]))
            affine_warping_dict[img_name] = list(matrix.reshape((6,)))
        json.dump(affine_warping_dict, fp, indent=2)

    with open(os.path.join(out_dir, 'skews.csv'), 'w') as fp:
        fp.write(info_txt)

    if os.path.exists(os.path.join(out_dir, 'perspective_images')):
        os.unlink(os.path.join(out_dir, 'perspective_images'))
    # os.symlink(perspective_img_dir, os.path.join(out_dir, 'perspective_images'))
    os.symlink(os.path.relpath(perspective_img_dir, out_dir),
               os.path.join(out_dir, 'perspective_images'))

    if os.path.exists(os.path.join(out_dir, 'perspective_dict.json')):
        os.unlink(os.path.join(out_dir, 'perspective_dict.json'))
    # os.symlink(perspective_file, os.path.join(out_dir, 'perspective_dict.json'))
    os.symlink(os.path.relpath(perspective_file, out_dir),
               os.path.join(out_dir, 'perspective_dict.json'))


def add_skew_to_pinhole_tracks(sparse_dir, warping_file):
    all_tracks = extract_all_tracks(sparse_dir)

    # read affine transformations
    inv_affine_warpings = {}
    with open(warping_file) as fp:
        affine_warpings = json.load(fp)
    for img_name in sorted(affine_warpings.keys()):
        matrix = np.array(affine_warpings[img_name]).reshape((2, 3))
        matrix = np.vstack((matrix, np.array([0, 0, 1]).reshape(1, 3)))
        matrix = np.linalg.inv(matrix)[0:2, :]
        inv_affine_warpings[img_name] = matrix

    cnt = len(all_tracks)
    for i in range(cnt):
        track = all_tracks[i]

        pixels = track['pixels']
        pixel_cnt = len(pixels)
        for j in range(pixel_cnt):
            img_name, col, row = pixels[j]

            # inv affine warp
            tmp = np.dot(inv_affine_warpings[img_name], np.array([col, row, 1.]).reshape((3, 1)))
            col = tmp[0, 0]
            row = tmp[1, 0]

            # modify all_tracks
            all_tracks[i]['pixels'][j] = (img_name, col, row)

    return all_tracks


if __name__ == '__main__':
    pass
