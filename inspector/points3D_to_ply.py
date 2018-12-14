import colmap.read_model as read_model
import sys
import numpy as np
from lib.plyfile import PlyData, PlyElement


def points3D_to_ply(points3D_file, ply_file):
    points = read_model.read_points3d_binary(points3D_file)
    points = [tuple(points[key].xyz) + tuple(points[key].rgb) for key in points]
    vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'uint8'), ('green', 'uint8'), ('blue', 'uint8')])


    el = PlyElement.describe(vertex, 'vertex')
    PlyData([el], byte_order='<').write(ply_file)


if __name__ == '__main__':
    points3D_file = sys.argv[1]
    ply_file = sys.argv[2]

    logging.info('points3D_file: {}'.format(points3D_file))
    logging.info('ply_file: {}'.format(ply_file))

    points3D_to_ply(points3D_file, ply_file)