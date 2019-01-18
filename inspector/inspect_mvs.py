import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
from colmap.read_dense import read_array
import os
import shutil
import tarfile
from colmap.extract_sfm import extract_sfm
import quaternion
from inspector.plot_reproj_err import plot_reproj_err


def visualize_depth_maps(colmap_dir):
    depth_dir = os.path.join(colmap_dir, 'dense/stereo/depth_maps')
    image_dir = os.path.join(colmap_dir, 'dense/images')

    inspect_dir = os.path.join(colmap_dir, 'dense/inspect')
    if os.path.exists(inspect_dir):
        shutil.rmtree(inspect_dir, ignore_errors=True)
    os.mkdir(inspect_dir)

    for img_name in os.listdir(image_dir):
        # copy
        shutil.copy(os.path.join(image_dir, img_name), inspect_dir)

        # check photometric
        # photometric_depth = os.path.join(depth_dir, '{}.photometric.bin'.format(img_name))
        # if os.path.exists(photometric_depth):
        #     depth_map = read_array(photometric_depth)
        #
        #     # Visualize the depth map.
        #     plt.figure()
        #
        #
        #     # min_depth, max_depth = np.percentile(depth_map, [5, 95])
        #     # # min_depth = 7.4
        #     # depth_map[depth_map < min_depth] = min_depth
        #     # depth_map[depth_map > max_depth] = max_depth
        #
        #     plt.imshow(depth_map, cmap='magma')
        #     plt.colorbar()
        #
        #     # plt.show()
        #     plt.savefig(os.path.join(inspect_dir, '{}.photometric.depth.jpg'.format(img_name)))
        #     plt.close()

        geometric_depth = os.path.join(depth_dir, '{}.geometric.bin'.format(img_name))
        if os.path.exists(geometric_depth):
            depth_map = read_array(geometric_depth)

            # Visualize the depth map.
            plt.figure()
            # min_depth, max_depth = np.percentile(depth_map, [5, 95])
            # # min_depth = 7.4
            # depth_map[depth_map < min_depth] = min_depth
            # depth_map[depth_map > max_depth] = max_depth

            # compute a mask
            mask = depth_map > 1e-10

            tmp = depth_map[mask]
            if tmp.size > 0:
                min_depth, max_depth = np.percentile(tmp, [5, 95])

                depth_map[depth_map < min_depth] = min_depth
                depth_map[depth_map > max_depth] = max_depth

            plt.imshow(depth_map, cmap='magma')
            plt.colorbar()

            # plt.show()
            plt.savefig(os.path.join(inspect_dir, '{}.geometric.depth.jpg'.format(img_name)))
            plt.close()

def check_reproj_error(camera_params, all_tracks):
    # construct projection matrix for each camera
    proj_matrices = {}
    for img_name in camera_params.keys():
        img_size, intrinsic, qvec, tvec = camera_params[img_name]
        K = np.array([[intrinsic[0], 0., intrinsic[2]],
                      [ 0., intrinsic[1], intrinsic[3]],
                      [ 0., 0., 1.]])
        R = quaternion.as_rotation_matrix(np.quaternion(qvec[0], qvec[1], qvec[2], qvec[3]))
        t = np.reshape(tvec, (3, 1))

        proj_matrices[img_name] = np.dot(K, np.hstack((R, t)))

    my_errs = []
    colmap_errs = []
    for track in all_tracks:
        xyz = np.array([track['xyz'][0], track['xyz'][1], track['xyz'][2], 1.]).reshape((4, 1))
        err = 0.
        for pixel in track['pixels']:
            img_name, col, row = pixel
            tmp = np.dot(proj_matrices[img_name], xyz)
            esti_col = tmp[0] / tmp[2]
            esti_row = tmp[1] / tmp[2]

            sq_err = (col - esti_col) ** 2 + (row - esti_row) ** 2
            err += np.sqrt(sq_err)
        err /= len(track['pixels'])
        # check whether it agrees with what colmap computes
        # if np.abs(err - track['err']) > 1e-1:
        #     print(track)
        #     print('err: {}'.format(err))
        #     exit(-1)

        my_errs.append(err)
        colmap_errs.append(track['err'])

    return np.array(my_errs), np.array(colmap_errs)


def vector_angle(vec1, vec2):
    vec1 = vec1 / np.sqrt(np.dot(vec1.T, vec1))
    vec2 = vec2 / np.sqrt(np.dot(vec2.T, vec2))

    tmp = np.dot(vec1.T, vec2)
    angle = np.rad2deg(np.math.acos(tmp))

    return angle


def inspect_input_sfm(colmap_dir):
    camera_params, all_tracks = extract_sfm('{}/dense/sparse'.format(colmap_dir))

    inspect_dir = '{colmap_dir}/dense/inspect'.format(colmap_dir=colmap_dir)

    my_errs, colmap_errs = check_reproj_error(camera_params, all_tracks)
    plot_reproj_err(my_errs, os.path.join(inspect_dir, 'sfm_reproj_err_my.jpg'))
    plot_reproj_err(colmap_errs, os.path.join(inspect_dir, 'sfm_reproj_err_colmap.jpg'))

    # save track to file
    with open(os.path.join(inspect_dir, 'sfm_feature_tracks'), 'w') as fp:
        fp.write('# format: x, y, z, reproj. err., track_length, img_name, col, row, ...\n')
        for track in all_tracks:
            line = '{} {} {} {} {}'.format(track['xyz'][0], track['xyz'][1], track['xyz'][2],
                                         track['err'], len(track['pixels']))
            for pixel in track['pixels']:
                img_name, col, row = pixel
                line += ' {} {} {}'.format(img_name, col, row)
            line += '\n'
            fp.write(line)

    with open(os.path.join(inspect_dir, 'camera_parameters.txt'), 'w') as fp:
        fp.write('# format: img_name, img_width, img_height, fx, fy, cx, cy, qw, qx, qy, qz, tx, ty, tz\n')

        for img_name in sorted(camera_params.keys()):
            img_size, intrinsic, qvec, tvec = camera_params[img_name]
            fp.write('{} {} {} {} {} {} {} {} {} {} {} {} {} {}\n'.format(img_name, img_size[0], img_size[1],
                                                                          intrinsic[0], intrinsic[1], intrinsic[2], intrinsic[3],
                                                                          qvec[0], qvec[1], qvec[2], qvec[3],
                                                                          tvec[0], tvec[1], tvec[2]))
        fp.write('\n')

    cam_center_positions = []
    img_center_rays = []

    img_leftright_angles = []
    img_updown_angles = []

    img_angle_variations = []

    for img_name in sorted(camera_params.keys()):
        img_size, intrinsic, qvec, tvec = camera_params[img_name]
        R = quaternion.as_rotation_matrix(np.quaternion(qvec[0], qvec[1], qvec[2], qvec[3]))
        t = np.reshape(tvec, (3, 1))
        cam_center_positions.append(np.dot(R.T, -t))

        K = np.array([[intrinsic[0], 0., intrinsic[2]],
                      [ 0., intrinsic[1], intrinsic[3]],
                      [ 0., 0., 1.]])
        width, height = img_size

        p1 = np.dot(np.linalg.inv(K), np.array([width/2., height/2., 1.]).reshape(3, 1))
        p2 = np.dot(np.linalg.inv(K), np.array([width/2., height/2., 10000.]).reshape(3, 1))
        img_center_rays.append(np.dot(R.T, p1-t) - np.dot(R.T, p2-t))

        left = np.dot(np.linalg.inv(K), np.array([0., height/2., 1.]).reshape(3, 1))
        right = np.dot(np.linalg.inv(K), np.array([width, height/2., 1.]).reshape(3, 1))
        img_leftright_angles.append(vector_angle(left, right))

        up = np.dot(np.linalg.inv(K), np.array([width/2., 0, 1.]).reshape(3, 1))
        down = np.dot(np.linalg.inv(K), np.array([width/2., height, 1.]).reshape(3, 1))
        img_updown_angles.append(vector_angle(up, down))

        print('left-right: {}'.format(img_leftright_angles[-1]))
        print('up-down: {}'.format(img_updown_angles[-1]))

        img_angle_variations.append(max([img_leftright_angles[-1], img_updown_angles[-1]]))

    # compute pair-wise angles
    cnt = len(cam_center_positions)
    print('{}: {}'.format(colmap_dir, cnt))
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
    plt.savefig(os.path.join(inspect_dir, 'field_of_view.jpg'))
    plt.close()

    # convert to unit vector
    # cam_center_positions = [x / np.sqrt(np.dot(x.T, x)) for x in cam_center_positions]
    # img_center_rays = [x / np.sqrt(np.dot(x.T, x)) for x in img_center_rays]

    cam_center_angles = np.zeros((cnt, cnt))
    img_center_angles = np.zeros((cnt, cnt))
    for i in range(cnt):
        for j in range(i+1, cnt):
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
    plt.savefig(os.path.join(inspect_dir, 'camera_center_pairwise_angles.jpg'))
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
    plt.savefig(os.path.join(inspect_dir, 'image_center_pairwise_angles.jpg'))
    plt.close()


def inspect_mvs(colmap_dir):
    # visualize_depth_maps(colmap_dir)

    # inspect mvs camera poses
    inspect_input_sfm(colmap_dir)

    # create a tar for the inspect_dir
    inspect_dir = os.path.join(colmap_dir, 'dense/inspect')
    with tarfile.open(os.path.join(colmap_dir, 'dense/inspect.tar'), 'w') as tar:
        tar.add(inspect_dir, arcname=os.path.basename(inspect_dir))


if __name__ == '__main__':
    #colmap_dir = '/data2/kz298/core3d_result/aoi-d1-wpafb/colmap'
    #colmap_dir = '/data2/kz298/core3d_result/aoi-d2-wpafb/colmap'
    #colmap_dir = '/data2/kz298/core3d_result/aoi-d3-ucsd/colmap'
    #colmap_dir = '/data2/kz298/core3d_result/aoi-d4-jacksonville/colmap'

    # colmap_dir = '/data2/kz298/mvs3dm_result/Failed_Explorer/colmap'

    # colmap_dir = '/data2/kz298/mvs3dm_result/Succeed_Explorer/colmap'

    mvs3dm_dir = '/data2/kz298/mvs3dm_result/'
    colmap_dirs = [os.path.join(mvs3dm_dir, item, 'colmap') for item in os.listdir(mvs3dm_dir)]

    for colmap_dir in colmap_dirs:
        print(colmap_dir)
        inspect_mvs(colmap_dir)

