import os
import numpy as np
import quaternion
import colmap.read_model as read_model
from lib.ransac import esti_simiarity


def read_data(colmap_dir):
    init_dir = os.path.join(colmap_dir, 'init')

    init_cam_positions = {}

    with open(os.path.join(init_dir, 'images.txt')) as fp:
        lines = [l.strip() for l in fp.readlines()]
        # remove empty lines
        lines = [l.split() for l in lines if l]

        for l in lines:
            qvec = np.array([float(x) for x in l[1:5]])
            tvec = np.array([float(x) for x in l[5:8]]).reshape((-1, 1))
            R = quaternion.as_rotation_matrix(np.quaternion(qvec[0], qvec[1], qvec[2], qvec[3]))
            init_cam_positions[int(l[0])] = -np.dot(R.T, tvec)

    sparse_ba_dir = os.path.join(colmap_dir, 'sparse_ba')

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
        source.append(ba_cam_positions[id].reshape((3, )))
        target.append(init_cam_positions[id].reshape((3, )))

    source = np.array(source)
    target = np.array(target)

    return source, target

def compute_transform(colmap_dir):
    source, target = read_data(colmap_dir)

    c, R, t = esti_simiarity(source, target, thres=1000)

    return c, R, t


if __name__ == '__main__':
    colmap_dir = '/data2/kz298/core3d_aoi/aoi-d4-jacksonville/colmap'

    c, R, t = compute_transform(colmap_dir)

    #logging.info('c: {}, R: {}, t: {}'.format(c, R, t))