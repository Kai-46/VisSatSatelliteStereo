import numpy as np
from lib.ply_np_converter import np2ply, ply2np
import json


# register the dense model via a linear mapping
def georegister_dense(in_ply, out_ply, aoi_json, M, t, filter=False):
    dense = ply2np(in_ply)

    points = dense[:, 0:3]
    normals = dense[:, 3:6]
    colors = dense[:, 6:9]

    # register
    points = np.dot(points, M) + np.tile(t, (points.shape[0], 1))
    normals = np.dot(normals, M) + np.tile(t, (normals.shape[0], 1))

    dense = np.hstack((points, normals, colors))

    with open(aoi_json) as fp:
        aoi_dict = json.load(fp)
    if filter:
        margin = 3 # meters
        east_min = aoi_dict['x'] - margin
        east_max = aoi_dict['x'] + aoi_dict['w'] + margin
        north_min = aoi_dict['y'] - aoi_dict['h'] - margin
        north_max = aoi_dict['y'] + margin

        mask = points[:, 0] > east_min
        mask = np.logical_and(mask, points[:, 0] < east_max)
        mask = np.logical_and(mask, points[:, 1] > north_min)
        mask = np.logical_and(mask, points[:, 1] < north_max)

        dense = dense[mask, :]

    comment_1 = 'projection: UTM {}{}'.format(aoi_dict['zone_number'], aoi_dict['zone_letter'])
    comment_2 = 'aoi bbx, x, y, w, h : {}, {}, {}, {}'.format(aoi_dict['x'], aoi_dict['y'], aoi_dict['w'], aoi_dict['h'])
    comments = [comment_1, comment_2]

    np2ply(dense, out_ply, comments)

    bbx = {}
    bbx['zone_number'] = aoi_dict['zone_number']
    bbx['zone_letter'] = aoi_dict['zone_letter']
    bbx['east_min'] = np.min(points[:, 0])
    bbx['east_max'] = np.max(points[:, 0])
    bbx['north_min'] = np.min(points[:, 1])
    bbx['north_max'] = np.max(points[:, 1])
    bbx['height_min'] = np.min(points[:, 2])
    bbx['height_max'] = np.max(points[:, 2])
    return bbx