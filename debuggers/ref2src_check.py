import os
import json
import numpy as np
import cv2
import random


def draw_circle(img, cols, rows):
    all_pts = []
    for i in range(len(cols)):
        col = cols[i]
        row = rows[i]
        f = cv2.KeyPoint()
        f.pt = (col, row)
        # Dummy size
        f.size = 10
        f.angle = 0
        f.response = 10

        all_pts.append(f)

    GREEN = (0, 255, 0)
    cv2.drawKeypoints(img, all_pts, img, color=GREEN)

    return img


def ref2src_check(work_dir):
    mvs_results_dir = os.path.join(work_dir, 'mvs_results')
    img_grid_dir = os.path.join(work_dir, 'mvs_results/height_maps/img_grid')
    img_grid_npy_dir = os.path.join(work_dir, 'mvs_results/height_maps/img_grid_npy')

    out_dir = os.path.join(mvs_results_dir, 'ref2src')
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    with open(os.path.join(mvs_results_dir, 'proj_mats.json')) as fp:
        proj_mats = json.load(fp)
    for img_name in proj_mats:
        proj_mats[img_name] = np.array(proj_mats[img_name]).reshape((4, 4))

    with open(os.path.join(mvs_results_dir, 'inv_proj_mats.json')) as fp:
        inv_proj_mats = json.load(fp)
    for img_name in proj_mats:
        inv_proj_mats[img_name] = np.array(inv_proj_mats[img_name]).reshape((4, 4))

    with open(os.path.join(mvs_results_dir, 'ref2src.json')) as fp:
        ref2src_dict = json.load(fp)

    for ref in sorted(ref2src_dict.keys()):
        print('processing ref {}...'.format(ref))

        out_subdir = os.path.join(out_dir, ref[:-4])
        if not os.path.exists(out_subdir):
            os.mkdir(out_subdir)

        # read ref image
        ref_img = cv2.imread(os.path.join(img_grid_dir, ref))

        num_pts = 2000
        cols = random.sample(range(ref_img.shape[1]), num_pts)
        rows = random.sample(range(ref_img.shape[0]), num_pts)

        ref_img = draw_circle(ref_img, cols, rows)

        cv2.imwrite(os.path.join(out_subdir, 'ref_' + ref), ref_img)

        # read the height map for ref image
        height_map = np.load(os.path.join(img_grid_npy_dir, ref + '.geometric.height.npy'))

        # now warp them into source images
        for src in sorted(ref2src_dict[ref]['src_img_names']):
            # concatenate inv_proj_mat with proj_mat
            P = np.dot(proj_mats[src], inv_proj_mats[ref])
            # loop
            src_cols = []
            src_rows = []
            for i in range(len(cols)):
                col = cols[i]
                row = rows[i]
                height = height_map[row, col]

                tmp = np.dot(P, np.array([col, row, 1.0, height]).reshape((4, 1)))

                src_col = tmp[0, 0] / tmp[2, 0]
                src_row = tmp[1, 0] / tmp[2, 0]

                src_cols.append(src_col)
                src_rows.append(src_row)

            src_img = cv2.imread(os.path.join(img_grid_dir, src))
            src_img = draw_circle(src_img, src_cols, src_rows)
            cv2.imwrite(os.path.join(out_subdir, 'src_' + src), src_img)


if __name__ == '__main__':
    work_dir = '/data2/kz298/core3d_result/aoi-d4-jacksonville'
    ref2src_check(work_dir)
