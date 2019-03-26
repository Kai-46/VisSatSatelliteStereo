import json
import os
from lib.rpc_model import RPCModel
import numpy as np
from lib.compose_proj_mat import compose_proj_mat


def project_sparse(inspect_dir, work_dir):
    metas_subdir = os.path.join(work_dir, 'metas/')
    rpc_models = {}
    for item in sorted(os.listdir(metas_subdir)):
        img_name = item[:-5] + '.png'
        with open(os.path.join(metas_subdir, item)) as fp:
            rpc_models[img_name] = RPCModel(json.load(fp))

    # with open(os.path.join(inspect_dir, 'kai_cameras.json')) as fp:
    #     cameras = json.load(fp)

    # with open(os.path.join(work_dir, 'colmap/sfm_perspective/init_camera_dict.json')) as fp:
    #     cameras = json.load(fp)

    with open(os.path.join(work_dir, 'approx_perspective_utm.json')) as fp:
        cameras = json.load(fp)

    for img_name in cameras:
        params = cameras[img_name][2:]
        cameras[img_name] = compose_proj_mat(params)

    local_points = np.loadtxt(os.path.join(inspect_dir, 'kai_coordinates.txt'))
    latlon_points = np.loadtxt(os.path.join(inspect_dir, 'kai_coordinates_latlon.txt'))

    track_file = os.path.join(inspect_dir, 'kai_tracks.json')
    with open(track_file) as fp:
        tracks = json.load(fp)

    fp = open(os.path.join(inspect_dir, 'kai_tracks.txt'), 'w')
    pixels = []
    rpc_pixels = []
    perspective_pixels = []
    for i in range(len(tracks)):
        track = tracks[i]
        lat = latlon_points[i, 0]
        lon = latlon_points[i, 1]
        height = latlon_points[i, 2]

        xx = local_points[i, 0]
        yy = local_points[i, 1]
        zz = local_points[i, 2]
        for img_name, col, row in track:
            pixels.append([col, row])
            fp.write('{} {} {} {}\n'.format(i, img_name, col, row))

            rpc_col, rpc_row = rpc_models[img_name].projection(lat, lon, height)
            rpc_pixels.append([rpc_col, rpc_row])

            P = cameras[img_name]
            tmp = np.dot(P, np.array([xx, yy, zz, 1.0]).reshape((4, 1)))
            perspective_col = tmp[0] / tmp[2]
            perspective_row = tmp[1] / tmp[2]
            perspective_pixels.append([perspective_col, perspective_row])

    fp.close()

    # try to compute error
    pixels = np.array(pixels)
    rpc_pixels = np.array(rpc_pixels)
    perspective_pixels = np.array(perspective_pixels)

    np.savetxt(os.path.join(inspect_dir, 'rpc_pixels.txt'), rpc_pixels)
    np.savetxt(os.path.join(inspect_dir, 'perspective_pixels.txt'), perspective_pixels)

    diff = np.sqrt(np.sum((pixels - rpc_pixels) ** 2, axis=1))

    np.savetxt(os.path.join(inspect_dir, 'diff.txt'), diff)

    tmp = np.percentile(diff, [70, 90])
    print('diff, min: {}, max: {}, median: {}, mean: {}, 70 percentile: {}, 90 percentile: {}'.format(np.min(diff), np.max(diff),
                                                                               np.median(diff), np.mean(diff),
                                                                               tmp[0], tmp[1]))


if __name__ == '__main__':
    work_dir = '/data2/kz298/mvs3dm_result/MasterSequesteredPark/'
    inspect_dir = os.path.join(work_dir, 'colmap/sfm_perspective/init_ba_triangulate_inspect')
    project_sparse(inspect_dir, work_dir)
