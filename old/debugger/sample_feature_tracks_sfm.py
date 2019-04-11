import numpy as np
import cv2
import os
import json


def draw_circle(img, col, row):
    f = cv2.KeyPoint()
    f.pt = (col, row)
    # Dummy size
    f.size = 5
    f.angle = 0
    f.response = 10

    GREEN = (0, 255, 0)
    cv2.drawKeypoints(img, [f, ], img, color=GREEN)

    return img


def sample_feature_tracks(sfm_dir, inspect_dir, out_dir, indices):
    image_dir = os.path.join(sfm_dir, 'images')
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    with open(os.path.join(inspect_dir, 'kai_tracks.json')) as fp:
        tracks = json.load(fp)

    for i in range(len(indices)):
        idx = indices[i]
        cur_track = tracks[idx]
        num_images = len(cur_track)

        img_per_row = 5
        if num_images < img_per_row:
            img_per_row = num_images + 1

        row_cnt = int(num_images / img_per_row) + 1

        img_size = 100
        gap = 2
        big_img = np.zeros((img_size * row_cnt + gap * (row_cnt - 1),
                            img_size * img_per_row + gap * (img_per_row - 1), 3))
        for j in range(num_images):
            # where to put in the big image
            big_img_row_idx = int(j / img_per_row) * (gap + img_size)
            big_img_col_idx = (j % img_per_row) * (gap + img_size)

            img_name, col, row = cur_track[j]

            img = cv2.imread(os.path.join(image_dir, img_name))
            img = draw_circle(img, col, row)
            # crop img
            col_min = int(col - img_size / 2)
            if col_min < 0:
                col_min = 0
            row_min = int(row - img_size / 2)
            if row_min < 0:
                row_min = 0
            img = img[row_min:row_min+img_size, col_min:col_min+img_size]

            big_img[big_img_row_idx:big_img_row_idx+img_size, big_img_col_idx:big_img_col_idx+img_size] = img

        cv2.imwrite(os.path.join(out_dir, 'track_{}.jpg'.format(i)), np.uint8(big_img))


if __name__ == '__main__':
    tile_dir = '/data2/kz298/mvs3dm_result/MasterSequesteredPark'
    sfm_dir = os.path.join(tile_dir, 'colmap/sfm_perspective/')
    inspect_dir = os.path.join(tile_dir, 'colmap/sfm_perspective/init_triangulate_ba_inspect')
    out_dir = os.path.join(inspect_dir, 'sample_tracks')

    cnt = 10000
    num_tracks_to_draw = 10
    indices = []
    for i in range(num_tracks_to_draw):
        idx = np.random.randint(cnt)
        indices.append(idx)

    sample_feature_tracks(sfm_dir, inspect_dir, out_dir,  indices)
