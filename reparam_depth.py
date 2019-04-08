from colmap.read_model import read_model
import numpy as np
from pyquaternion import Quaternion
import os


def robust_depth_range(depth_range):
    for img_name in depth_range:
        if depth_range[img_name]:
            tmp = depth_range[img_name]
            max_val = max(tmp)
            min_val = min(tmp)
            print('img_name: {}, depth min: {}, max: {}, ratio: {}'.format(img_name, min_val,
                                                                max_val, max_val / min_val))
            tmp = sorted(depth_range[img_name])
            cnt = len(tmp)
            min_depth = tmp[int(0.01 * cnt)]
            max_depth = tmp[int(0.99 * cnt)]

            lower_stretch = 10
            upper_stretch = 30
            min_depth_new = min_depth - lower_stretch
            max_depth_new = max_depth + upper_stretch
            if max_depth_new <= min_depth_new:
                min_depth_new = min_depth
                max_depth_new = max_depth

            depth_range[img_name] = (min_depth_new, max_depth_new)
        else:
            depth_range[img_name] = (-1e20, -1e20)

    return depth_range


# e.g. last_row=[0, 0, 1, 0] represents the plane z=0
# the first three elements should be a unit vector
def reparam_depth(sparse_dir, save_dir):
    # this method is documented in colmap src/mvs/model.cc
    colmap_cameras, colmap_images, colmap_points3D = read_model(sparse_dir, ext='.txt')

    depth_range = {}
    for img_id in colmap_images:
        img_name = colmap_images[img_id].name
        depth_range[img_name] = []

    z_values = []
    for point3D_id in colmap_points3D:
        point3D = colmap_points3D[point3D_id]
        x = point3D.xyz.reshape((3, 1))
        z_values.append(x[2, 0])
        for img_id in point3D.image_ids:
            img_name = colmap_images[img_id].name
            qvec = colmap_images[img_id].qvec
            tvec = colmap_images[img_id].tvec.reshape((3, 1))
            R = Quaternion(qvec[0], qvec[1], qvec[2], qvec[3]).rotation_matrix
            x1 = np.dot(R, x) + tvec  # do not change x
            depth = x1[2, 0]
            if depth > 0:
                depth_range[img_name].append(depth)

    depth_range = robust_depth_range(depth_range)

    # protective margin 20 meters
    margin = 20.0
    min_z_value = np.percentile(z_values, 1) - margin
    print('min_z_value: {}'.format(min_z_value))
    z_values = None

    # reparametrize depth
    last_row = np.array([0., 0., 1., -min_z_value]).reshape((1, 4))
    last_rows = {}

    reparam_depth_range = {}
    for img_id in colmap_images:
        img_name = colmap_images[img_id].name
        reparam_depth_range[img_name] = []

    common_reparam_depth_range = []

    for point3D_id in colmap_points3D:
        point3D = colmap_points3D[point3D_id]
        x = point3D.xyz.reshape((3, 1))
        depth = 0
        for img_id in point3D.image_ids:
            img_name = colmap_images[img_id].name
            qvec = colmap_images[img_id].qvec
            tvec = colmap_images[img_id].tvec.reshape((3, 1))
            R = Quaternion(qvec[0], qvec[1], qvec[2], qvec[3]).rotation_matrix

            cam_id = colmap_images[img_id].camera_id
            fx, fy, cx, cy = colmap_cameras[cam_id].params
            K = np.array([[fx, 0., cx],
                          [0., fy, cy],
                          [0., 0., 1.]])
            P_3by4 = np.dot(K, np.hstack((R, tvec)))

            depth_min = depth_range[img_name][0]
            depth_max = depth_range[img_name][1]
            print('depth_min: {}, depth_max: {}, ratio: {}'.format(depth_min, depth_max, depth_max / depth_min))
            P_4by4 = np.vstack((P_3by4, depth_min * last_row))

            if img_name not in last_rows:
                last_rows[img_name] = depth_min * last_row

            x1 = np.vstack((x, np.array([[1.,]])))
            tmp = np.dot(P_4by4, x1)
            # depth is the fourth component, instead of its inverse
            depth = tmp[3, 0] / tmp[2, 0]
            if depth > 0:
                # depth_new = np.dot(last_row, x1)
                # assert (np.abs(depth - depth_new) < 2.0)

                reparam_depth_range[img_name].append(depth)

        if depth > 0:
            common_reparam_depth_range.append(depth)

    reparam_depth_range = robust_depth_range(reparam_depth_range)

    # save to file
    with open(os.path.join(save_dir, 'raw_depth.txt'), 'w') as fp:
        fp.write('# format: img_name, depth_min, depth_max\n')
        for img_name in sorted(depth_range.keys()):
            min_depth, max_depth = depth_range[img_name]
            fp.write('{} {} {}\n'.format(img_name, min_depth, max_depth))

    with open(os.path.join(save_dir, 'reparam_depth.txt'), 'w') as fp:
        fp.write('# format: img_name, depth_min, depth_max\n')
        for img_name in sorted(reparam_depth_range.keys()):
            min_depth, max_depth = reparam_depth_range[img_name]
            fp.write('{} {} {}\n'.format(img_name, min_depth, max_depth))

    with open(os.path.join(save_dir, 'last_rows.txt'), 'w') as fp:
        for img_name in sorted(last_rows.keys()):
            vec = last_rows[img_name]
            fp.write('{} {} {} {} {}\n'.format(img_name, vec[0, 0], vec[0, 1], vec[0, 2], vec[0, 3]))

    with open(os.path.join(save_dir, 'reference_plane.txt'), 'w') as fp:
        fp.write('{} {} {} {}\n'.format(last_row[0, 0], last_row[0, 1], last_row[0, 2], last_row[0, 3]))

    common_reparam_depth_range = sorted(common_reparam_depth_range)
    cnt = len(common_reparam_depth_range)
    lower_stretch = 10
    upper_stretch = 100.
    min_depth = common_reparam_depth_range[int(0.01 * cnt)] - lower_stretch
    max_depth = common_reparam_depth_range[int(0.99 * cnt)] + upper_stretch
    print('{} points, depth_min: {}, depth_max: {}'.format(cnt, min_depth, max_depth))

    with open(os.path.join(save_dir, 'reparam_depth_range.txt'), 'w') as fp:
        fp.write('{} {}\n'.format(min_depth, max_depth))


if __name__ == '__main__':
    mvs_dir = '/data2/kz298/mvs3dm_result/MasterSequesteredPark/colmap/mvs'
    reparam_depth(os.path.join(mvs_dir, 'sparse'), mvs_dir)
