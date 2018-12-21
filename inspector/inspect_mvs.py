import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
from colmap.read_dense import read_array
import os
import shutil

def inspect_mvs(colmap_dir):
    depth_dir = os.path.join(colmap_dir, 'dense/stereo/depth_maps')
    image_dir = os.path.join(colmap_dir, 'dense/images')

    inspect_dir = os.path.join(colmap_dir, 'dense/inspect')
    if not os.path.exists(inspect_dir):
        os.mkdir(inspect_dir)

    for img_name in os.listdir(image_dir):
        # copy
        shutil.copy(os.path.join(image_dir, img_name), inspect_dir)

        # check photometric
        photometric_depth = os.path.join(depth_dir, '{}.photometric.bin'.format(img_name))
        if os.path.exists(photometric_depth):
            depth_map = read_array(photometric_depth)

            # Visualize the depth map.
            plt.figure()
            min_depth, max_depth = np.percentile(depth_map, [5, 95])
            # min_depth = 7.4
            depth_map[depth_map < min_depth] = min_depth
            depth_map[depth_map > max_depth] = max_depth
            plt.imshow(depth_map, cmap='magma')
            plt.colorbar()

            # plt.show()
            plt.savefig(os.path.join(inspect_dir, '{}.photometric.depth.jpg'.format(img_name)))
            plt.close()

        geometric_depth = os.path.join(depth_dir, '{}.geometric.bin'.format(img_name))
        if os.path.exists(geometric_depth):
            depth_map = read_array(geometric_depth)

            # Visualize the depth map.
            plt.figure()
            min_depth, max_depth = np.percentile(depth_map, [5, 95])
            # min_depth = 7.4
            depth_map[depth_map < min_depth] = min_depth
            depth_map[depth_map > max_depth] = max_depth
            plt.imshow(depth_map, cmap='magma')
            plt.colorbar()

            # plt.show()
            plt.savefig(os.path.join(inspect_dir, '{}.geometric.depth.jpg'.format(img_name)))
            plt.close()


if __name__ == '__main__':
    colmap_dir = '/data2/kz298/core3d_result/aoi-d1-wpafb/colmap'

    #colmap_dir = '/data2/kz298/core3d_aoi/aoi-d4-jacksonville/colmap'
    inspect_mvs(colmap_dir)
