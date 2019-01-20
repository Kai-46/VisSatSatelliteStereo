from colmap.read_model import read_model
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import numpy as np
import quaternion
from inspector.plot_reproj_err import plot_reproj_err
from colmap.extract_sfm import read_tracks, read_camera_params
from inspector.check_reproj_error import check_reproj_error
from inspector.vector_angle import vector_angle
from lib.ply_np_converter import np2ply


def inspect_sfm(colmap_dir):
    inspect_dir = os.path.join(colmap_dir, 'inspect')
    if not os.path.exists(inspect_dir):
        os.mkdir(inspect_dir)

    sparse_dir = os.path.join(colmap_dir, 'sfm_perspective/sparse')
    out_dir = os.path.join(inspect_dir)
    sfm_inspector = SparseInspector(sparse_dir, out_dir, 'PERSPECTIVE')
    sfm_inspector.inspect_depth_range()

    sparse_dir = os.path.join(colmap_dir, 'sfm_perspective/sparse_ba')
    out_dir = os.path.join(inspect_dir, 'sfm_perspective')
    sfm_inspector = SparseInspector(sparse_dir, out_dir, 'PERSPECTIVE')
    sfm_inspector.inspect_all()

    sparse_dir = os.path.join(colmap_dir, 'sfm_pinhole/sparse_ba')
    out_dir = os.path.join(inspect_dir, 'sfm_pinhole')
    sfm_inspector = SparseInspector(sparse_dir, out_dir, 'PINHOLE')
    sfm_inspector.inspect_all()


class SparseInspector(object):
    def __init__(self, sparse_dir, out_dir, camera_model, ext='.txt'):
        assert (camera_model == 'PINHOLE' or camera_model == 'PERSPECTIVE')
        self.camera_model = camera_model

        self.sparse_dir = os.path.abspath(sparse_dir)
        self.out_dir = os.path.abspath(out_dir)

        if not os.path.exists(self.out_dir):
            os.mkdir(self.out_dir)

        self.cameras, self.images, self.points3D = read_model(self.sparse_dir, ext)
        self.camera_params = read_camera_params(self.cameras, self.images)
        self.all_tracks = read_tracks(self.images, self.points3D)

        self.img_cnt = len(self.images.keys())


    def inspect_all(self):
        self.inspect_reproj_err()
        self.inspect_image_key_points()
        self.inspect_depth_range()
        self.inspect_scene_points()
        self.inspect_angles()
        self.inspect_tracks()

    def inspect_image_key_points(self):
        # img_id2name = []
        img_names = []
        img_widths = []
        img_heights = []

        key_point_cnt = []
        for image_id in self.images:
            image = self.images[image_id]

            # img_id2name.append((image_id, image.name))
            img_names.append(image.name)
            key_point_cnt.append(len(image.xys))

            cam = self.cameras[image.camera_id]
            img_widths.append(cam.width)
            img_heights.append(cam.height)

        # with open(os.path.join(self.out_dir, 'inspect_img_id2name.json'), 'w') as fp:
        #     json.dump(img_id2name, fp)

        plt.figure()
        plt.bar(range(0, self.img_cnt), key_point_cnt)
        plt.xticks(ticks=range(0, self.img_cnt), labels=img_names, rotation=90)
        plt.ylabel('# of sift features')
        plt.grid(True)
        plt.title('total # of images: {}'.format(self.img_cnt))
        plt.tight_layout()

        plt.savefig(os.path.join(self.out_dir, 'inspect_key_points.jpg'))
        plt.close()
        #plt.show()

        plt.figure()
        plt.plot(range(0, self.img_cnt), img_widths, 'b-o', label='width')
        #plt.legend('width')
        plt.plot(range(0, self.img_cnt), img_heights, 'r-+', label='height')
        #plt.legend('height')
        plt.legend()
        #plt.legend('width', 'height')
        plt.xticks(ticks=range(0, self.img_cnt), labels=img_names, rotation=90)
        plt.ylabel('# of pixels')
        plt.grid(True)
        plt.title('total # of images: {}'.format(self.img_cnt))
        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, 'inspect_image_size.jpg'))
        plt.close()

    # this method is documented in colmap src/mvs/model.cc
    def inspect_depth_range(self):
        depth_range = {}
        for img_id in self.images:
            img_name = self.images[img_id].name
            depth_range[img_name] = []

        for point3D_id in self.points3D:
            point3D = self.points3D[point3D_id]
            x = point3D.xyz.reshape((3, 1))
            for img_id in point3D.image_ids:
                img_name = self.images[img_id].name
                qvec = self.images[img_id].qvec
                tvec = self.images[img_id].tvec.reshape((3, 1))
                R = quaternion.as_rotation_matrix(np.quaternion(qvec[0], qvec[1], qvec[2], qvec[3]))
                x = np.dot(R,x) + tvec
                depth = x[2, 0]
                if depth > 0:
                    depth_range[img_name].append(depth)

        for img_name in depth_range:
            if depth_range[img_name]:
                tmp = sorted(depth_range[img_name])
                cnt = len(tmp)
                min_depth = tmp[int(0.01 * cnt)] * (1 - 0.25)
                max_depth = tmp[int(0.99 * cnt)] * (1 + 0.25)
                depth_range[img_name] = (min_depth, max_depth)
            else:
                depth_range[img_name] = (0, 0)

        with open(os.path.join(self.out_dir, 'inspect_depth_range.txt'), 'w') as fp:
            fp.write('# format: img_name, min_depth, max_depth\n')
            for img_name in sorted(depth_range.keys()):
                min_depth, max_depth = depth_range[img_name]
                fp.write('{} {} {}\n'.format(img_name, min_depth, max_depth))
            fp.write('\n')

    def inspect_reproj_err(self):
        my_errs, colmap_errs = check_reproj_error(self.camera_params, self.all_tracks, self.camera_model)
        plot_reproj_err(my_errs, os.path.join(self.out_dir, 'inspect_reproj_err_my.jpg'))
        plot_reproj_err(colmap_errs, os.path.join(self.out_dir, 'inspect_reproj_err_colmap.jpg'))

    def inspect_tracks(self):
        # save camera params and all_tracks
        # save track to file
        with open(os.path.join(self.out_dir, 'sfm_feature_tracks'), 'w') as fp:
            fp.write('# format: x, y, z, reproj. err., track_length, img_name, col, row, ...\n')
            for track in self.all_tracks:
                line = '{} {} {} {} {}'.format(track['xyz'][0], track['xyz'][1], track['xyz'][2],
                                               track['err'], len(track['pixels']))
                for pixel in track['pixels']:
                    img_name, col, row = pixel
                    line += ' {} {} {}'.format(img_name, col, row)
                line += '\n'
                fp.write(line)

        with open(os.path.join(self.out_dir, 'camera_parameters.txt'), 'w') as fp:
            if self.camera_model == 'PINHOLE':
                fp.write('# format: img_name, img_width, img_height, fx, fy, cx, cy, qw, qx, qy, qz, tx, ty, tz\n')

                for img_name in sorted(self.camera_params.keys()):
                    img_size, intrinsic, qvec, tvec = self.camera_params[img_name]
                    fp.write('{} {} {} {} {} {} {} {} {} {} {} {} {} {}\n'.format(img_name, img_size[0], img_size[1],
                                                                                  intrinsic[0], intrinsic[1], intrinsic[2],
                                                                                  intrinsic[3],
                                                                                  qvec[0], qvec[1], qvec[2], qvec[3],
                                                                                  tvec[0], tvec[1], tvec[2]))
            else:
                fp.write('# format: img_name, img_width, img_height, fx, fy, cx, cy, s, qw, qx, qy, qz, tx, ty, tz\n')

                for img_name in sorted(self.camera_params.keys()):
                    img_size, intrinsic, qvec, tvec = self.camera_params[img_name]
                    fp.write('{} {} {} {} {} {} {} {} {} {} {} {} {} {} {}\n'.format(img_name, img_size[0], img_size[1],
                                                                                  intrinsic[0], intrinsic[1],
                                                                                  intrinsic[2],
                                                                                  intrinsic[3], intrinsic[4],
                                                                                  qvec[0], qvec[1], qvec[2], qvec[3],
                                                                                  tvec[0], tvec[1], tvec[2]))
            fp.write('\n')

        # check distribution of track_len
        track_len = np.array([len(track['pixels']) for track in self.all_tracks])
        plt.figure(figsize=(14, 5), dpi=80)
        max_track_len = np.max(track_len)
        plt.hist(track_len, bins=np.arange(0.5, max_track_len + 1.5, 1))
        plt.xticks(range(1, max_track_len+1))
        plt.ylabel('# of tracks')
        plt.xlabel('track length')
        plt.title('total # of images: {}\ntotal # of tracks: {}\ntrack length, min: {}, mean: {:.6f}, median: {}, max: {}'
                  .format(self.img_cnt, len(self.all_tracks), np.min(track_len), np.mean(track_len), np.median(track_len), max_track_len))
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, 'inspect_track_len.jpg'))
        plt.close()

    def inspect_scene_points(self):
        points = np.array([tuple(self.points3D[id].xyz) for id in self.points3D])

        np2ply(points, os.path.join(self.out_dir, 'sfm_scene_points.ply'))

    def inspect_angles(self):
        cam_center_positions = []
        img_center_rays = []

        img_leftright_angles = []
        img_updown_angles = []

        img_angle_variations = []

        for img_name in sorted(self.camera_params.keys()):
            img_size, intrinsic, qvec, tvec = self.camera_params[img_name]
            R = quaternion.as_rotation_matrix(np.quaternion(qvec[0], qvec[1], qvec[2], qvec[3]))
            t = np.reshape(tvec, (3, 1))
            cam_center_positions.append(np.dot(R.T, -t))

            if self.camera_model == 'PINHOLE':
                K = np.array([[intrinsic[0], 0., intrinsic[2]],
                              [0., intrinsic[1], intrinsic[3]],
                              [0., 0., 1.]])
            else:
                K = np.array([[intrinsic[0], intrinsic[4], intrinsic[2]],
                              [0., intrinsic[1], intrinsic[3]],
                              [0., 0., 1.]])
            width, height = img_size

            p1 = np.dot(np.linalg.inv(K), np.array([width / 2., height / 2., 1.]).reshape(3, 1))
            p2 = np.dot(np.linalg.inv(K), np.array([width / 2., height / 2., 10000.]).reshape(3, 1))
            img_center_rays.append(np.dot(R.T, p1 - t) - np.dot(R.T, p2 - t))

            left = np.dot(np.linalg.inv(K), np.array([0., height / 2., 1.]).reshape(3, 1))
            right = np.dot(np.linalg.inv(K), np.array([width, height / 2., 1.]).reshape(3, 1))
            img_leftright_angles.append(vector_angle(left, right))

            up = np.dot(np.linalg.inv(K), np.array([width / 2., 0, 1.]).reshape(3, 1))
            down = np.dot(np.linalg.inv(K), np.array([width / 2., height, 1.]).reshape(3, 1))
            img_updown_angles.append(vector_angle(up, down))

            print('left-right: {}'.format(img_leftright_angles[-1]))
            print('up-down: {}'.format(img_updown_angles[-1]))

            img_angle_variations.append(max([img_leftright_angles[-1], img_updown_angles[-1]]))

        # compute pair-wise angles
        cnt = len(cam_center_positions)
        plt.figure(figsize=(14, 8))
        plt.plot(range(0, cnt), img_angle_variations)
        # plt.plot(range(0, cnt), img_leftright_angles)
        # plt.plot(range(0, cnt), img_updown_angles)
        # plt.legend(['max', 'left-right', 'up-down'])
        plt.xticks(range(0, cnt, 1))
        plt.xlabel('image index')
        plt.grid(True)
        plt.ylabel('field of view (degrees)')
        plt.title('field of view')
        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, 'field_of_view.jpg'))
        plt.close()

        # convert to unit vector
        # cam_center_positions = [x / np.sqrt(np.dot(x.T, x)) for x in cam_center_positions]
        # img_center_rays = [x / np.sqrt(np.dot(x.T, x)) for x in img_center_rays]

        cam_center_angles = np.zeros((cnt, cnt))
        img_center_angles = np.zeros((cnt, cnt))
        for i in range(cnt):
            for j in range(i + 1, cnt):
                cam_center_angles[i, j] = vector_angle(cam_center_positions[i], cam_center_positions[j])
                img_center_angles[i, j] = vector_angle(img_center_rays[i], img_center_rays[j])

        # for visualization purposesd
        mask = cam_center_angles < 1e-10
        cam_center_angles[cam_center_angles < 1e-10] = np.min(cam_center_angles[np.logical_not(mask)]) - 5

        mask = img_center_angles < 1e-10
        img_center_angles[img_center_angles < 1e-10] = np.min(img_center_angles[np.logical_not(mask)]) - 1

        plt.figure(figsize=(14, 10))
        plt.imshow(cam_center_angles, cmap='magma')
        plt.colorbar()

        # plt.show()
        plt.xticks(range(0, cnt, 1))
        plt.xlabel('image index')
        plt.yticks(range(0, cnt, 1))
        plt.ylabel('image index')
        plt.title('camera_center_pairwise_angles (degrees)')
        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, 'camera_center_pairwise_angles.jpg'))
        plt.close()

        plt.figure(figsize=(14, 10))
        plt.imshow(img_center_angles, cmap='magma')
        plt.colorbar()

        # plt.show()
        plt.xticks(range(0, cnt, 1))
        plt.xlabel('image index')
        plt.yticks(range(0, cnt, 1))
        plt.ylabel('image index')
        plt.title('image_center_pairwise_angles (degrees)')
        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, 'image_center_pairwise_angles.jpg'))
        plt.close()


if __name__ == '__main__':
    # mvs3dm_dir = '/data2/kz298/mvs3dm_result/'
    # colmap_dirs = [os.path.join(mvs3dm_dir, item, 'colmap') for item in os.listdir(mvs3dm_dir)]

    colmap_dirs = ['/data2/kz298/mvs3dm_result/Explorer/colmap']


    for colmap_dir in colmap_dirs:
        print(colmap_dir)

        inspect_sfm(colmap_dir)

        # inspect_dir = os.path.join(colmap_dir, 'inspect')
        # if not os.path.exists(inspect_dir):
        #     os.mkdir(inspect_dir)
        #
        # sparse_dir = os.path.join(colmap_dir, 'sfm_perspective/sparse_ba')
        # if not os.path.exists(sparse_dir):
        #     continue
        # out_dir = os.path.join(inspect_dir, 'sfm_perspective')
        #
        # sfm_inspector = SparseInspector(sparse_dir, out_dir, 'PERSPECTIVE')
        # sfm_inspector.inspect_reproj_err()
        # sfm_inspector.inspect_image_key_points()
        # sfm_inspector.inspect_depth_range()
        # sfm_inspector.inspect_scene_points()
        # sfm_inspector.inspect_angles()
        # sfm_inspector.inspect_tracks()
        #
        # sparse_dir = os.path.join(colmap_dir, 'sfm_pinhole/sparse_ba')
        # out_dir = os.path.join(inspect_dir, 'sfm_pinhole')
        #
        # sfm_inspector = SparseInspector(sparse_dir, out_dir, 'PINHOLE')
        # sfm_inspector.inspect_reproj_err()
        # sfm_inspector.inspect_image_key_points()
        # sfm_inspector.inspect_depth_range()
        # sfm_inspector.inspect_scene_points()
        # sfm_inspector.inspect_angles()
        # sfm_inspector.inspect_tracks()