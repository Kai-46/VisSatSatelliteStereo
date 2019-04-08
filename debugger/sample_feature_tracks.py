import numpy as np
import cv2
import os


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


def sample_feature_tracks(tile_dir, dsm_pixels_dir, out_dir, indices):
    image_dir = os.path.join(tile_dir, 'images')
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    image_names = sorted(os.listdir(image_dir))
    num_images = len(image_names)

    img_per_row = 5
    row_cnt = int(num_images / img_per_row) + 1
    img_size = 100
    gap = 2
    big_img = np.zeros((img_size * row_cnt + gap * (row_cnt - 1),
                        img_size * img_per_row + gap * (img_per_row - 1), 3))

    for i in range(len(indices)):
        idx = indices[i]
        for j in range(num_images):
            # where to put in the big image
            big_img_row_idx = int(j / img_per_row) * (gap + img_size)
            big_img_col_idx = (j % img_per_row) * (gap + img_size)

            pixels = np.load(os.path.join(dsm_pixels_dir, image_names[j][:-4] + '.npy'))
            col = pixels[idx, 0]
            row = pixels[idx, 1]

            img = cv2.imread(os.path.join(image_dir, image_names[j]))
            img = draw_circle(img, col, row)
            # crop img
            col_min = int(col - img_size / 2)
            row_min = int(row - img_size / 2)
            img = img[row_min:row_min+img_size, col_min:col_min+img_size]

            big_img[big_img_row_idx:big_img_row_idx+img_size, big_img_col_idx:big_img_col_idx+img_size] = img

        cv2.imwrite(os.path.join(out_dir, 'track_{}.jpg'.format(i)), np.uint8(big_img))


if __name__ == '__main__':
    tile_dir = '/data2/kz298/mvs3dm_result/MasterSequesteredPark'

    cnt = 500000
    num_tracks_to_draw = 10
    indices = []
    for i in range(num_tracks_to_draw):
        idx = np.random.randint(cnt)
        indices.append(idx)

    dsm_pixels_dir = os.path.join(tile_dir, 'ground_truth/dsm_pixels_rpc')
    out_dir = os.path.join(tile_dir, 'ground_truth/sample_tracks_rpc')
    sample_feature_tracks(tile_dir, dsm_pixels_dir, out_dir, indices)

    dsm_pixels_dir = os.path.join(tile_dir, 'ground_truth/dsm_pixels_init')
    out_dir = os.path.join(tile_dir, 'ground_truth/sample_tracks_init')
    sample_feature_tracks(tile_dir, dsm_pixels_dir, out_dir, indices)

    dsm_pixels_dir = os.path.join(tile_dir, 'ground_truth/dsm_pixels_final')
    out_dir = os.path.join(tile_dir, 'ground_truth/sample_tracks_final')
    sample_feature_tracks(tile_dir, dsm_pixels_dir, out_dir, indices)
