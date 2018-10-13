import imageio
import numpy as np
import os
import shutil


def calibrate(tile_dir):
    dest_dir = os.path.join(tile_dir, 'calibrate')
    if os.path.exists(dest_dir):
        shutil.rmtree(dest_dir, ignore_errors=True)
    os.mkdir(dest_dir)

    item_list = sorted(os.listdir(os.path.join(tile_dir, 'images')))
    cnt = len(item_list)
    # compute mean of the first
    im = imageio.imread(os.path.join(tile_dir, 'images', item_list[0]))
    mean_val = np.mean(im)
    imageio.imwrite(os.path.join(dest_dir, item_list[0]), im)

    for i in range(1, cnt):
        print('calibrating the image {}/{}'.format(i, cnt))
        im = imageio.imread(os.path.join(tile_dir, 'images', item_list[i])).astype(dtype=np.float)
        # adjust mean
        tmp = mean_val - np.mean(im)
        im += tmp

        im[im < 0] = 0
        im[im > 255] = 255
        imageio.imwrite(os.path.join(dest_dir, item_list[i]), im.astype(dtype=np.uint8))


if __name__ == '__main__':
    tile_dir = 'data_500'
    calibrate(tile_dir)
