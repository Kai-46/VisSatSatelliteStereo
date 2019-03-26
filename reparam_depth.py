from colmap.read_model import read_model
import numpy as np
from pyquaternion import Quaternion
import os


def robust_depth_range(depth_range):
    for img_name in depth_range:
        if depth_range[img_name]:
            tmp = sorted(depth_range[img_name])
            cnt = len(tmp)
            min_depth = tmp[int(0.02 * cnt)]
            max_depth = tmp[int(0.98 * cnt)]

            stretch = 5
            min_depth_new = min_depth - stretch
            max_depth_new = max_depth + stretch
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
            x1 = np.dot(R,x) + tvec # do not change x
            depth = x1[2, 0]
            if depth > 0:
                depth_range[img_name].append(depth)

    depth_range = robust_depth_range(depth_range)

    # protective margin 20 meters
    min_z_value = np.percentile(z_values, 1) - 20
    print('min_z_value: {}'.format(min_z_value))
    z_values = None

    # reparametrize depth
    last_row = np.array([0., 0., 1., -min_z_value]).reshape((1, 4))
    last_rows = {}

    reparam_depth_range = {}
    for img_id in colmap_images:
        img_name = colmap_images[img_id].name
        reparam_depth_range[img_name] = []

    for point3D_id in colmap_points3D:
        point3D = colmap_points3D[point3D_id]
        x = point3D.xyz.reshape((3, 1))
        # print('height: {}'.format(x[2]))
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
            P_4by4 = np.vstack((P_3by4, depth_min * last_row))

            if img_name not in last_rows:
                last_rows[img_name] = depth_min * last_row

            x1 = np.vstack((x, np.array([[1.,]])))
            x1 = np.dot(P_4by4, x1)
            # depth is the fourth component, instead of its inverse
            depth = x1[3, 0] / x1[2, 0]

            reparam_depth_range[img_name].append(depth)

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


if __name__ == '__main__':
    # sparse_dir = '/data2/kz298/mvs3dm_result/Explorer/colmap/sfm_pinhole/init_triangulate_ba'
    sparse_dir = '/data2/kz298/core3d_result/aoi-d4-jacksonville/colmap/sparse_for_mvs'
    save_dir = sparse_dir
    reparam_depth(sparse_dir, save_dir)
