import os
import quaternion
import numpy as np
import colmap.read_model as read_model
from lib.ransac import esti_simiarity


def read_data(proj_dir):
    init_dir = os.path.join(proj_dir, 'init')

    init_cam_positions = {}

    with open(os.path.join(init_dir, 'images.txt')) as fp:
        lines = [l.strip() for l in fp.readlines()]
        # remove empty lines
        lines = [l.split() for l in lines if not l]

        for l in lines:
            qvec = np.array([float(x) for x in lines[1:5]])
            tvec = np.array([float(x) for x in lines[5:8]]).reshape((-1, 1))
            R = quaternion.as_rotation_matrix(quaternion.from_float_array(qvec))
            init_cam_positions[int(l[0])] = -np.dot(R.T, tvec)

    sparse_ba_dir = os.path.join(proj_dir, 'sparse_ba')

    ba_cam_positions = {}
    _, images, _ = read_model.read_model(sparse_ba_dir, '.bin')
    for img_id in images:
        img = images[img_id]
        qvec = img.qvec
        tvec = img.tvec.reshape((-1, 1))
        R = quaternion.as_rotation_matrix(quaternion.from_float_array(qvec))
        ba_cam_positions[img_id] = -np.dot(R.T, tvec)

    # re-organize the array
    source = []
    target = []
    for id in init_cam_positions:
        source.append(ba_cam_positions[id].reshape((1, 3)))
        target.append(init_cam_positions[id].reshape((1, 3)))

    source = np.array(source)
    target = np.array(target)

    return source, target

def compute_transform(colmap_dir):
    source, target = read_data(colmap_dir)

    c, R, t = esti_simiarity(source, target)

    return c, R, t


