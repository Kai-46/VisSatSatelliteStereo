import numpy as np
from lib.ply_np_converter import np2ply, ply2np
import json


# register the dense model via a linear mapping
def georegister_dense(in_ply, out_ply, aoi_json, M, t, filter=False):
    dense = ply2np(in_ply)

    points = dense[:, 0:3]

    has_normals = (dense.shape[1] > 3)

    if has_normals:
        normals = dense[:, 3:6]

    has_colors = (dense.shape[1] > 6)
    if has_colors:
        colors = dense[:, 6:9]

    # register
    points = np.dot(points, M) + np.tile(t, (points.shape[0], 1))

    if has_normals:
        normals = np.dot(normals, M) + np.tile(t, (normals.shape[0], 1))

    if has_colors:
        dense = np.hstack((points, normals, colors))
    elif has_normals:
        dense = np.hstack((points, normals))
    else:
        dense = points


    with open(aoi_json) as fp:
        aoi_dict = json.load(fp)
    if filter:
        margin = 3 # meters
        east_min = aoi_dict['ul_easting'] - margin
        east_max = aoi_dict['ul_easting'] + aoi_dict['width'] + margin
        north_min = aoi_dict['ul_northing'] - aoi_dict['height'] - margin
        north_max = aoi_dict['ul_northing'] + margin

        mask = points[:, 0] > east_min
        mask = np.logical_and(mask, points[:, 0] < east_max)
        mask = np.logical_and(mask, points[:, 1] > north_min)
        mask = np.logical_and(mask, points[:, 1] < north_max)

        dense = dense[mask, :]

    comment_1 = 'projection: UTM {}{}'.format(aoi_dict['zone_number'], aoi_dict['hemisphere'])
    comment_2 = 'aoi bbx, x, y, w, h : {}, {}, {}, {}'.format(aoi_dict['ul_easting'], aoi_dict['ul_northing'], aoi_dict['width'], aoi_dict['height'])
    comments = [comment_1, comment_2]

    np2ply(dense, out_ply, comments)

    bbx = {}
    bbx['zone_number'] = aoi_dict['zone_number']
    bbx['hemisphere'] = aoi_dict['hemisphere']
    bbx['east_min'] = np.min(points[:, 0])
    bbx['east_max'] = np.max(points[:, 0])
    bbx['north_min'] = np.min(points[:, 1])
    bbx['north_max'] = np.max(points[:, 1])
    bbx['height_min'] = np.min(points[:, 2])
    bbx['height_max'] = np.max(points[:, 2])
    return bbx
