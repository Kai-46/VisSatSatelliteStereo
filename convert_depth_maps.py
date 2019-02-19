import os
import numpy as np
import shutil
from colmap.read_dense import read_array
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import imageio


def save_image_only(matrix, save_file):
    fig = plt.figure()
    dpi = fig.get_dpi()
    fig.set_size_inches(matrix.shape[1] / dpi, matrix.shape[0] / dpi)

    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.axis('off')
    fig.add_axes(ax)

    ax.imshow(matrix, cmap='magma')

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    w, h = fig.canvas.get_width_height()
    im = data.reshape((h, w, 3)).astype(dtype=np.uint8)
    plt.close(fig)

    imageio.imwrite(save_file, im)


def convert_depth_maps(mvs_dir, out_dir):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # first load inv_proj_mats
    inv_proj_mats = {}
    with open(os.path.join(mvs_dir, 'inv_proj_mats.txt')) as fp:
        for line in fp.readlines():
            tmp = line.split(' ')
            img_name = tmp[0]
            mats = np.array([float(tmp[i]) for i in range(1, 17)]).reshape((4, 4))
            inv_proj_mats[img_name] = mats

    # then read the depth maps
    depth_dir = os.path.join(mvs_dir, 'stereo/depth_maps')
    image_dir = os.path.join(mvs_dir, 'images')
    for item in sorted(os.listdir(depth_dir)):
        #depth_type = 'photometric'

        depth_type = 'geometric'
        idx = item.rfind('.{}.bin'.format(depth_type))
        if idx == -1:
            continue

        img_name = item[:idx]

        # copy raw image
        shutil.copy(os.path.join(image_dir, img_name), out_dir)

        depth_map = read_array(os.path.join(depth_dir, item))
        # create a meshgrid
        height, width = depth_map.shape
        col, row = np.meshgrid(range(width), range(height))

        col = col.reshape((1, -1))
        row = row.reshape((1, -1))

        depth = depth_map[row, col]

        depth = depth.reshape((1, -1))

        mask = depth <= 0

        tmp = np.vstack((col, row, np.ones((1, width * height)), 1.0 / depth))
        tmp = np.dot(inv_proj_mats[img_name], tmp)

        tmp[0, :] /= tmp[3, :]
        tmp[1, :] /= tmp[3, :]
        tmp[2, :] /= tmp[3, :]

        # tmp[2, :]

        # disp_depth_min = 0
        # disp_depth_max = 50

        disp_depth_min = 10
        disp_depth_max = 50

        tmp[2:3, :][mask] = disp_depth_min  # set height to zero

        print('{}, emtpy ratio: {}%'.format(img_name, np.sum(mask) / mask.size * 100))
        height_map = tmp[2:3, :].reshape((height, width))

        # Visualize the depth map.
        # plt.figure()

        # compute a mask
        # min_height, max_height = np.percentile(height_map[height_map>0], [4, 95])
        # height_map[height_map < min_height] = min_height - 5
        # height_map[height_map > max_height] = max_height + 5

        # truncate
        height_map[height_map > disp_depth_max] = disp_depth_max
        height_map[height_map < disp_depth_min] = disp_depth_min

        # plt.imshow(height_map, cmap='magma')

        # plt.colorbar()
        # # plt.show()
        # plt.savefig(os.path.join(out_dir, '{}.{}.height.jpg'.format(img_name, depth_type)))
        # plt.close()

        # plt.axis('off')
        # plt.savefig(os.path.join(out_dir, '{}.{}.height.jpg'.format(img_name, depth_type)), bbox_inches=0)

        save_image_only(height_map, os.path.join(out_dir, '{}.{}.height.jpg'.format(img_name, depth_type)))


def convert_normal_maps(mvs_dir, out_dir):
    # normal_type = 'photometric'
    normal_type = 'geometric'

    normal_dir = os.path.join(mvs_dir, 'stereo/normal_maps')

    for item in sorted(os.listdir(normal_dir)):
        idx = item.rfind('.{}.bin'.format(normal_type))
        if idx == -1:
            continue

        img_name = item[:idx]
        normal_file = os.path.join(normal_dir, '{}.{}.bin'.format(img_name, normal_type))

        normal_map = read_array(normal_file)

        # filter absurd value
        normal_map[normal_map < -1e19] = -1.0

        normal_map = (normal_map + 1.0) / 2

        # print('normal_map, min: {}, max: {}'.format(np.min(normal_map), np.max(normal_map)))

        # plt.figure()
        # plt.imshow(normal_map, cmap='magma')
        #
        # # plt.show()
        # plt.savefig(os.path.join(out_dir, '{}.{}.normal.jpg'.format(img_name, normal_type)))
        # plt.close()

        save_image_only(normal_map, os.path.join(out_dir, '{}.{}.normal.jpg'.format(img_name, normal_type)))


if __name__ == '__main__':
    mvs_dir = '/data2/kz298/mvs3dm_result/MasterSequesteredPark/colmap/mvs'
    #mvs_dir = '/data2/kz298/core3d_result/aoi-d4-jacksonville/colmap/mvs'
    #mvs_dir = '/home/kai/satellite_stereo/Explorer_subset/colmap/mvs'
    out_dir = os.path.join(mvs_dir, 'height_maps')

    #convert_depth_maps(mvs_dir, out_dir)
    convert_normal_maps(mvs_dir, out_dir)
