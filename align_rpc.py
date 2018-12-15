from lib.rpc_triangulate import triangulate
from colmap.read_model import read_model
import os
import json
from lib.rpc_model import RPCModel
import numpy as np

from lib.ransac import esti_simiarity


# read tracks
# each track is dict
def read_tracks(colmap_dir):
    sparse_dir = os.path.join(colmap_dir, 'sparse_ba')

    colmap_cameras, colmap_images, colmap_points3D = read_model.read_model(sparse_dir, '.bin')

    all_tracks = []

    for point3D_id in colmap_points3D:
        point3D = colmap_points3D[point3D_id]
        image_ids = point3D.image_ids
        point2D_idxs = point3D.point2D_idxs

        cur_track = {}
        cur_track['xyz'] = (point3D.xyz[0], point3D.xyz[1], point3D.xyz[2])
        cur_track['err'] = point3D.error

        cur_track_len = len(image_ids)
        assert (cur_track_len == len(point2D_idxs))
        pixels = []
        for i in range(cur_track_len):
            image = colmap_images[image_ids[i]]
            img_name = image.name

            point2D_idx = point2D_idxs[i]
            point2D = image.xys[point2D_idx]
            assert (image.point3D_ids[point2D_idx] == point3D_id)

            pixels.append((img_name, point2D[0], point2D[1]))

        cur_track['pixels'] = pixels
        all_tracks.append(cur_track)

    return all_tracks

def read_data(work_dir):
    all_tracks = read_tracks(os.path.join(work_dir, 'colmap'))

    with open(os.path.join(work_dir, 'approx_perspective_utm.json')) as fp:
        perspective_dict = json.load(fp)

    for i in range(len(all_tracks)):
        for pixel in all_tracks[i]['pixels']:
            img_name, col, row = pixel

            params = perspective_dict['img_name']
            fy = params[1]
            s = params[4]
            norm_skew = s / fy
            col += norm_skew * row

            all_tracks[i]['pixels'] = (img_name, col, row)

    # now start to create all points
    with open(os.path.join(work_dir, 'approx_affine_latlon.json')) as fp:
        affine_dict = json.load(fp)

    source = []
    target = []
    for i in range(len(all_tracks)):
        source.append(all_tracks['xyz'])

        rpc_models = []
        affine_models = []
        track = []
        for pixel in all_tracks[i]['pixels']:
            img_name, col, row = pixel
            meta_file = os.path.join(work_dir, img_name[:img_name.rfind('.')] + '.json')
            with open(meta_file) as fp:
                meta_dict = json.load(fp)
            rpc_models.append(RPCModel(meta_dict))

            P = np.array(affine_dict[img_name]).reshape((2, 4))

            affine_models.append(P)

        _, final_point_utm = triangulate(track, rpc_models, affine_models)
        target.append([final_point_utm[0], final_point_utm[1], final_point_utm[2]])

    source = np.array(source)
    target = np.array(target)
    return source, target

def compute_transform(work_dir):
    source, target = read_data(work_dir)

    c, R, t = esti_simiarity(source, target)

    return c, R, t

if __name__ == '__main__':
    import logging
    import sys
    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    work_dir = '/data2/kz298/core3d_aoi/aoi-d4-jacksonville/'
    #work_dir = '/data2/kz298/core3d_aoi/aoi-d4-jacksonville-overlap/'

    c, R, t = compute_transform(work_dir)