from lib.rpc_triangulate import triangulate
from colmap.read_model import read_model
import os
import json
from lib.rpc_model import RPCModel
import numpy as np
import logging
from lib.esti_similarity import esti_similarity, esti_similarity_ransac
from lib.esti_linear import esti_linear
from inspector.plot_reproj_err import plot_reproj_err

# read tracks
# each track is dict
def read_tracks(colmap_dir):
    sparse_dir = os.path.join(colmap_dir, 'sparse_norm_ba')

    colmap_cameras, colmap_images, colmap_points3D = read_model(sparse_dir, '.txt')

    all_tracks = []

    for point3D_id in colmap_points3D:
        point3D = colmap_points3D[point3D_id]

        # only use small error tracks
        if point3D.error >= 1:
            continue

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

    # select a subset of all_tracks
    # if len(all_tracks) > 5000:
    #     all_tracks = all_tracks[:5000]

    # now start to create all points
    with open(os.path.join(work_dir, 'approx_affine_latlon.json')) as fp:
        affine_dict = json.load(fp)

    # load the common scene coordinate frame for perspective approx.
    with open(os.path.join(work_dir, 'aoi.json')) as fp:
        aoi_dict = json.load(fp)
    aoi_ul_east = aoi_dict['x']
    aoi_ul_north = aoi_dict['y']

    source = []
    target = []
    tmp_file = os.path.join(work_dir, 'tmpfile.txt')

    rpc_reproj_errs = []
    for i in range(len(all_tracks)):
        # if len(all_tracks[i]['pixels']) == 2: # ignore two-view tracks
        #     continue

        source.append(all_tracks[i]['xyz'])

        rpc_models = []
        affine_models = []
        track = []
        for pixel in all_tracks[i]['pixels']:
            img_name, col, row = pixel
            track.append((col, row))

            meta_file = os.path.join(work_dir, 'metas/{}.json'.format(img_name[:img_name.rfind('.')]))
            with open(meta_file) as fp:
                meta_dict = json.load(fp)
            rpc_models.append(RPCModel(meta_dict))
            # try modify the offset params
            #rpc_models[-1].rowOff

            P = np.array(affine_dict[img_name]).reshape((2, 4))

            affine_models.append(P)

        logging.info('triangulating {}/{} points, track length: {}'.format(i+1, len(all_tracks), len(track)))
        _, final_point_utm, err = triangulate(track, rpc_models, affine_models, tmp_file)

        # note that UTM coordinate is a left-handed coordinate system
        # change to the common scene coordinate frame
        # use a smaller number and change to the right-handed coordinate frame
        xx = aoi_ul_north - final_point_utm[1]
        yy = final_point_utm[0] - aoi_ul_east

        target.append([xx, yy, final_point_utm[2]])

        rpc_reproj_errs.append(err)

    # remove tmpfile.txt
    os.remove(tmp_file)

    # for debug
    # check reprojection error
    plot_reproj_err(rpc_reproj_errs, os.path.join(work_dir, 'inspect_rpc_reproj_err.jpg'))

    perspective_reproj_errs = [track['err'] for track in all_tracks]
    plot_reproj_err(perspective_reproj_errs, os.path.join(work_dir, 'inspect_perspective_reproj_err.jpg'))

    # remove target that has a bigger error
    # source_new = []
    # target_new = []
    # for i in range(len(rpc_reproj_errs)):
    #     if rpc_reproj_errs[i] < 1.:
    #         source_new.append(source[i])
    #         target_new.append(target[i])
    #
    # logging.info('\ntotal # of useful points: {}\n'.format(len(source_new)))

    # source = np.array(source_new)
    # target = np.array(target_new)

    source = np.array(source)
    target = np.array(target)

    return source, target
#
# def compute_transform(work_dir, use_ransac=False):
#     source, target = read_data(work_dir)
#
#     if use_ransac:
#         c, R, t = esti_similarity_ransac(source, target)
#     else:
#         c, R, t = esti_similarity(source, target)
#
#     return c, R, t


def compute_transform(work_dir, use_ransac=False):
    source, target = read_data(work_dir)

    M, t = esti_linear(source, target)

    return M, t


def test_align(work_dir):
    use_ransac = False

    # set log file
    log_file = os.path.join(work_dir, 'logs/log_align-sparse_norm_ba-to-rpc-linear.txt')
    # logging.basicConfig(filename=log_file, level=logging.INFO, filemode='w')
    log_hanlder = logging.FileHandler(log_file, 'w')
    log_hanlder.setLevel(logging.INFO)
    logging.root.setLevel(logging.INFO)
    logging.root.addHandler(log_hanlder)

    from datetime import datetime
    since = datetime.now()
    logging.info('Starting at {} ...'.format(since.strftime('%Y-%m-%d %H:%M:%S')))

    M, t= compute_transform(work_dir, use_ransac=use_ransac)

    ending = datetime.now()
    duration = (ending - since).total_seconds() / 60. # in minutes
    logging.info('Finishing at {}, duration: {}'.format(ending.strftime('%Y-%m-%d %H:%M:%S'), duration))

    # remove logging handler for later use
    logging.root.removeHandler(log_hanlder)


if __name__ == '__main__':
    # import sys
    # logging.getLogger().setLevel(logging.INFO)
    # logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    #work_dir = '/data2/kz298/core3d_aoi/aoi-d4-jacksonville/'
    #work_dir = '/data2/kz298/core3d_aoi/aoi-d4-jacksonville-overlap/'

    #work_dir = '/data2/kz298/core3d_result_bak/aoi-d1-wpafb/'
    #work_dir = '/data2/kz298/core3d_result_bak/aoi-d2-wpafb/'
    #work_dir = '/data2/kz298/core3d_result_bak/aoi-d3-ucsd/'
    #work_dir = '/data2/kz298/core3d_result_bak/aoi-d4-jacksonville/'

    #work_dir = '/data2/kz298/core3d_result/aoi-d1-wpafb/'
    #work_dir = '/data2/kz298/core3d_result/aoi-d2-wpafb/'
    #work_dir = '/data2/kz298/core3d_result/aoi-d3-ucsd/'
    #work_dir = '/data2/kz298/core3d_result/aoi-d4-jacksonville/'

    work_dirs = ['/data2/kz298/core3d_result/aoi-d1-wpafb/',
                 '/data2/kz298/core3d_result/aoi-d2-wpafb/',
                 '/data2/kz298/core3d_result/aoi-d3-ucsd/',
                 '/data2/kz298/core3d_result/aoi-d4-jacksonville/']

    for work_dir in work_dirs:
        test_align(work_dir)
