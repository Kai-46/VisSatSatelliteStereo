import numpy as np
import json
import logging
import os
import imageio
from lib.warp_affine import warp_affine
from colmap.extract_sfm import extract_all_tracks


def remove_skew(perspective_img_dir, perspective_file, pinhole_img_dir, pinhole_file, warping_file):
    if not os.path.exists(pinhole_img_dir):
        os.mkdir(pinhole_img_dir)

    with open(perspective_file) as fp:
        perspective_dict = json.load(fp)

    pinhole_dict = {}
    affine_warping_dict = {}
    for img_name in sorted(os.listdir(perspective_img_dir)):
        # w, h, fx, fy, cx, cy, s, qvec, t
        params = perspective_dict[img_name]
        w = params[0]
        h = params[1]
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

        affine_warping_dict[img_name] = affine_matrix

        new_h, new_w = img_dst.shape
        # add off_set to camera parameters
        cx += off_set[0]
        cy += off_set[1]

        logging.info('removed normalized skew: {} in image: {}, original size: {}, {}, new image size: {}, {}'.format(norm_skew, img_name, w, h, new_w, new_h))

        pinhole_dict[img_name] = [new_w, new_h, fx, fy, cx, cy,
                                  qvec[0], qvec[1], qvec[2], qvec[3],
                                  tvec[0], tvec[1], tvec[2]]

    with open(pinhole_file, 'w') as fp:
        json.dump(pinhole_dict, fp, indent=2, sort_keys=True)

    with open(warping_file, 'w') as fp:
        for img_name in sorted(affine_warping_dict.keys()):
            matrix = affine_warping_dict[img_name]
            # fp.write('{} {} {} {} {} {} {}\n'.format(img_name, matrix[0, 0], matrix[0, 1], matrix[0, 2],
            #                                          matrix[1, 0], matrix[1, 1], matrix[1,2]))
            affine_warping_dict[img_name] = list(matrix.reshape((6, )))
        json.dump(affine_warping_dict, fp, indent=2, sort_keys=True)


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