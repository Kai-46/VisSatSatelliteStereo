import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from colmap.read_dense import read_array
import os
import shutil


def inspect_mvs(colmap_dir):
    inspect_dir = os.path.join(colmap_dir, 'inspect')
    if not os.path.exists(inspect_dir):
        os.mkdir(inspect_dir)

    dense_dir = os.path.join(colmap_dir, 'mvs')
    out_dir = os.path.join(inspect_dir, 'mvs')
    mvs_inspector = DenseInspector(dense_dir, out_dir)
    mvs_inspector.inspect_depth_maps()
    mvs_inspector.inspect_normal_maps()


class DenseInspector(object):
    def __init__(self, dense_dir, out_dir):
        self.dense_dir = dense_dir
        self.out_dir = out_dir

        if not os.path.exists(self.out_dir):
            os.mkdir(self.out_dir)

    def inspect_depth_maps(self):
        depth_dir = os.path.join(self.dense_dir, 'stereo/depth_maps')
        image_dir = os.path.join(self.dense_dir, 'images')

        for img_name in os.listdir(image_dir):
            # copy raw image
            shutil.copy(os.path.join(image_dir, img_name), self.out_dir)

            depth_type = 'photometric'

            #depth_type = 'geometric'

            depth_file = os.path.join(depth_dir, '{}.{}.bin'.format(img_name, depth_type))
            if os.path.exists(depth_file):
                depth_map = read_array(depth_file)

                # Visualize the depth map.
                plt.figure()
                # min_depth, max_depth = np.percentile(depth_map, [5, 95])
                # # min_depth = 7.4
                # depth_map[depth_map < min_depth] = min_depth
                # depth_map[depth_map > max_depth] = max_depth

                # compute a mask
                mask = depth_map > -1e19

                tmp = depth_map[mask]
                if tmp.size > 0:
                    min_depth, max_depth = np.percentile(tmp, [5, 95])

                    depth_map[depth_map < min_depth] = min_depth
                    depth_map[depth_map > max_depth] = max_depth

                plt.imshow(depth_map, cmap='magma')
                plt.colorbar()

                # plt.show()
                plt.savefig(os.path.join(self.out_dir, '{}.{}.depth.jpg'.format(img_name, depth_type)))
                plt.close()

    def inspect_normal_maps(self):
        normal_dir = os.path.join(self.dense_dir, 'stereo/normal_maps')
        image_dir = os.path.join(self.dense_dir, 'images')

        for img_name in os.listdir(image_dir):
            # copy raw image
            # shutil.copy(os.path.join(image_dir, img_name), self.out_dir)

            normal_type = 'photometric'
            # normal_type = 'geometric'

            normal_file = os.path.join(normal_dir, '{}.{}.bin'.format(img_name, normal_type))
            if os.path.exists(normal_file):
                normal_map = read_array(normal_file)

                # filter absurd value
                normal_map[normal_map < -1e19] = -1.0

                normal_map = (normal_map + 1.0) / 2

                # print('normal_map, min: {}, max: {}'.format(np.min(normal_map), np.max(normal_map)))

                plt.figure()
                plt.imshow(normal_map, cmap='magma')
                # plt.colorbar()

                # plt.show()
                plt.savefig(os.path.join(self.out_dir, '{}.{}.normal.jpg'.format(img_name, normal_type)))
                plt.close()



if __name__ == '__main__':
    # work_dirs = ['/data2/kz298/mvs3dm_result/Explorer',
    #             '/data2/kz298/mvs3dm_result/MasterProvisional1',
    #             '/data2/kz298/mvs3dm_result/MasterProvisional2',
    #             '/data2/kz298/mvs3dm_result/MasterProvisional3',
    #             '/data2/kz298/mvs3dm_result/MasterSequestered1',
    #             '/data2/kz298/mvs3dm_result/MasterSequestered2',
    #             '/data2/kz298/mvs3dm_result/MasterSequestered3',
    #             '/data2/kz298/mvs3dm_result/MasterSequesteredPark']
    # colmap_dirs = [os.path.join(work_dir, 'colmap') for work_dir in work_dirs]
    # colmap_dirs = [colmap_dirs[0], ]
    #
    # for colmap_dir in colmap_dirs:
    #     print(colmap_dir)
    #     inspect_mvs(colmap_dir)

    #colmap_dir = '/home/kai/satellite_stereo/Explorer_subset/colmap'

    colmap_dir = '/data2/kz298/mvs3dm_result/MasterSequestered1/colmap'
    inspect_mvs(colmap_dir)
