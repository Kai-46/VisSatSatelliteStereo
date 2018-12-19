import numpy as np
from lib.plyfile import PlyData, PlyElement
import json
import logging


# register the dense model via a linear mapping
def georegister_dense(in_ply, out_ply, aoi_json, M, t):
    dense = PlyData.read(in_ply)

    points = np.hstack((dense['vertex']['x'].reshape((-1, 1)),
                        dense['vertex']['y'].reshape((-1, 1)),
                        dense['vertex']['z'].reshape((-1, 1))))
    normals = np.hstack((dense['vertex']['nx'].reshape((-1, 1)),
                         dense['vertex']['ny'].reshape((-1, 1)),
                         dense['vertex']['nz'].reshape((-1, 1))))
    colors = np.hstack((dense['vertex']['red'].reshape((-1, 1)),
                        dense['vertex']['green'].reshape((-1, 1)),
                        dense['vertex']['blue'].reshape((-1, 1))))

    points_reg = np.dot(points, M) + np.tile(t, (points.shape[0], 1))
    normals_reg = np.dot(normals, M) + np.tile(t, (normals.shape[0], 1))

    # z = points_reg[:, 2]
    # below_thres = np.percentile(z, 0)
    # above_thres = np.percentile(z, 100)
    # logging.info('below_thres: {}, above_thres: {}'.format(below_thres, above_thres))
    #
    # mask = np.logical_and(z>=below_thres, z<=above_thres)
    # #mask = np.tile(mask.reshape((-1, 1)), (1, 3))
    #
    # points_reg = points_reg[mask, :]
    # normals_reg = normals_reg[mask, :]
    # colors = colors[mask, :]

    # read json file
    # with open(os.path.join(proj_dir, 'roi.json')) as fp:
    #     roi = json.load(fp)
    
    with open(aoi_json) as fp:
        roi = json.load(fp)
    comment_1 = 'projection: UTM {}{}'.format(roi['zone_number'], roi['zone_letter'])
    comment_2 = 'x, y, w, h : {}, {}, {}, {}'.format(roi['x'], roi['y'], roi['w'], roi['h'])
    logging.info(comment_1)
    logging.info(comment_2)

    comments = [comment_1, comment_2]

    # write to plydata object
    # in perspective camera approximation, the world coordinate frame is (south, east, above)
    # the UTM coordinate frame is (east, north, above)
    x = points_reg[:, 1:2] + roi['x']
    y = roi['y'] - points_reg[:, 0:1]
    z = points_reg[:, 2:3]

    nx = normals_reg[:, 1:2]
    ny = normals_reg[:, 0:1]
    nz = normals_reg[:, 2:3]

    points = np.hstack((x, y, z, nx, ny, nz, colors))
    vertex = np.array([tuple(point) for point in points], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                                                                 ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
                                                                 ('red', 'uint8'), ('green', 'uint8'), ('blue', 'uint8')])
    el = PlyElement.describe(vertex, 'vertex')

    PlyData([el], byte_order='<', comments=comments).write(out_ply)
