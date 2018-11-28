import lib.read_model as read_model
import sys
import numpy as np
from lib.plyfile import PlyData, PlyElement
import json


def points3D_to_ply(points3D_file, roi_file, ply_file):
    points = read_model.read_points3d_binary(points3D_file)
    points = [tuple(points[key].xyz) + tuple(points[key].rgb) for key in points]
    vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'uint8'), ('green', 'uint8'), ('blue', 'uint8')])
    
    # read json file
    with open(roi_file) as fp:
        roi = json.load(fp)
    comment_1 = 'projection: UTM {}{}'.format(roi['zone_number'], roi['zone_letter'])
    comment_2 = 'x, y, w, h : {}, {}, {}, {}'.format(roi['x'], roi['y'], roi['w'], roi['h'])
    print(comment_1)
    print(comment_2)
    comments = [comment_1, comment_2]

    el = PlyElement.describe(vertex, 'vertex')
    PlyData([el], byte_order='<', comments=comments).write(ply_file)


if __name__ == '__main__':
    points3D_file = sys.argv[1]
    roi_file = sys.argv[2]
    ply_file = sys.argv[3]

    print('points3D_file: {}'.format(points3D_file))
    print('roi_file: {}'.format(roi_file))
    print('ply_file: {}'.format(ply_file))

    points3D_to_ply(points3D_file, roi_file, ply_file)