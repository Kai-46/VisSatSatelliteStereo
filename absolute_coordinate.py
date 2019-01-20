from lib.rpc_triangulate import triangulate
from colmap.read_model import read_model
import os
import json
from lib.rpc_model import RPCModel
import numpy as np
import logging
from colmap.extract_sfm import read_tracks
import multiprocessing
from lib.ply_np_converter import np2ply
from inspector.plot_reproj_err import plot_reproj_err


def read_sfm_pinhole(work_dir):
    sparse_dir = os.path.join(work_dir, 'colmap/sfm_pinhole/sparse_ba')
    _, colmap_images, colmap_points3D = read_model(sparse_dir, '.txt')
    all_tracks = read_tracks(colmap_images, colmap_points3D)

    # read affine transformations
    inv_affine_warpings = {}
    with open(os.path.join(work_dir, 'colmap/sfm_pinhole/affine_warpings.txt')) as fp:
        for line in fp.readlines():
            tmp = line.strip().split()
            img_name = tmp[0]
            matrix = np.array([float(x) for x in tmp[1:7]]).reshape((2, 3))
            matrix = np.vstack((matrix, np.array([0, 0, 1]).reshape(1, 3)))
            matrix = np.linalg.inv(matrix)[0:2, :]
            inv_affine_warpings[img_name] = matrix

    cnt = len(all_tracks)
    normalized = np.zeros((cnt, 4))
    pinhole_track_lines = ['# format: track_length, img_name, col, row, ...\n', ]
    perspective_track_lines = ['# format: track_length, img_name, col, row, ...\n', ]
    for i in range(cnt):
        track = all_tracks[i]
        xyz = track['xyz']
        normalized[i, 0] = xyz[0]
        normalized[i, 1] = xyz[1]
        normalized[i, 2] = xyz[2]
        normalized[i, 3] = track['err']

        pixels = track['pixels']
        pixel_cnt = len(pixels)
        pinhole_line = '{} '.format(pixel_cnt)
        perspective_line = '{} '.format(pixel_cnt)
        for j in range(pixel_cnt):
            img_name, col, row = pixels[j]
            pinhole_line += ' {} {} {}'.format(img_name, col, row)

            # inv affine warp
            tmp = np.dot(inv_affine_warpings[img_name], np.array([col, row, 1.]).reshape((3, 1)))
            col = tmp[0, 0]
            row = tmp[1, 0]
            perspective_line += ' {} {} {}'.format(img_name, col, row)

            # modify all_tracks
            all_tracks[i]['pixels'][j] = (img_name, col, row)
        pinhole_track_lines.append(pinhole_line + '\n')
        perspective_track_lines.append(perspective_line + '\n')

    np.savetxt(os.path.join(work_dir, 'register/normalized_coordinates.txt'), normalized,
               header='# format: x, y, z, reproj_err')
    with open(os.path.join(work_dir, 'register/tracks_in_sfm_pinhole.txt'), 'w') as fp:
        fp.writelines(pinhole_track_lines)

    with open(os.path.join(work_dir, 'register/tracks_in_sfm_perspective.txt'), 'w') as fp:
        fp.writelines(perspective_track_lines)

    # tracks
    all_tracks = [track['pixels'] for track in all_tracks]

    return all_tracks


def run_triangulation(result_file, work_dir, track_file, tmp_file):
    # load RPC model and affine approximation
    with open(os.path.join(work_dir, 'approx_affine_latlon.json')) as fp:
        affine_dict = json.load(fp)

    rpc_dict = {}
    for img_name in affine_dict:
        # projection matrix
        affine_dict[img_name] = np.array(affine_dict[img_name]).reshape((2, 4))

        meta_file = os.path.join(work_dir, 'metas/{}.json'.format(img_name[:-4]))
        with open(meta_file) as fp:
            rpc_dict[img_name] = RPCModel(json.load(fp))

    absolute = []
    with open(track_file) as fp:
        all_tracks = json.load(fp)
    cnt = len(all_tracks)

    pid = os.getpid()
    logging.info('running rpc triangulation process: {}, # of tracks: {}'.format(pid, cnt))

    for i in range(cnt):
        rpc_models = []
        affine_models = []
        track = []
        for pixel in all_tracks[i]:
            img_name, col, row = pixel
            track.append((col, row))
            rpc_models.append(rpc_dict[img_name])
            affine_models.append(affine_dict[img_name])

        _, final_point_utm, err = triangulate(track, rpc_models, affine_models, tmp_file)

        zone_letter = 1 if final_point_utm[4] == 'N' else -1
        absolute.append([final_point_utm[0], final_point_utm[1], final_point_utm[2], final_point_utm[3], zone_letter, err])

    # remove tmpfile.txt
    os.remove(tmp_file)

    # write to result_file
    np.save(result_file, absolute)

    logging.info('process {} done, triangulated {} points'.format(pid, cnt))


def triangualte_all_points(work_dir):
    register_subdir = os.path.join(work_dir, 'register')
    if not os.path.exists(register_subdir):
        os.mkdir(register_subdir)
    tmp_subdir = os.path.join(work_dir, 'tmp')
    if not os.path.exists(tmp_subdir):
        os.mkdir(tmp_subdir)

    all_tracks = read_sfm_pinhole(work_dir)

    process_cnt = 10
    processes = []
    # split all_tracks into multiple chunks
    chunk_size = int(len(all_tracks) / process_cnt)
    chunks = [[j*chunk_size, (j+1)*chunk_size] for j in range(process_cnt)]
    chunks[-1][1] = len(all_tracks)
    for i in range(process_cnt):
        track_file = os.path.join(work_dir, 'tmp/tracks_{}.txt'.format(i))
        with open(track_file, 'w') as fp:
            idx1 = chunks[i][0]
            idx2 = chunks[i][1]
            json.dump(all_tracks[idx1:idx2], fp)
        tmp_file = os.path.join(work_dir, 'tmp/tmpfile_{}.txt'.format(i))
        result_file = os.path.join(work_dir, 'tmp/absolute_{}.npy'.format(i))

        t = multiprocessing.Process(target=run_triangulation, args=(result_file, work_dir, track_file, tmp_file))
        processes.append(t)
        t.start()

    for one_process in processes:
        one_process.join()

    # read result_files
    absolute = []
    for i in range(process_cnt):
        absolute.append(np.load(os.path.join(work_dir, 'tmp/absolute_{}.npy'.format(i))))
    absolute = np.vstack(tuple(absolute))

    np.savetxt(os.path.join(work_dir, 'register/absolute_coordinates.txt'), absolute,
               header='# format: easting, northing, height, zone_number, zone_letter (1 for N and -1 for S), reproj_err')

    logging.info('RPC triangulation done, triangulated {} points in total'.format(len(all_tracks)))

    # add inspector
    zone_number = absolute[0, 3]
    zone_letter = 'N' if absolute[0, 4] > 0 else 'S'
    comments = ['projection: UTM {}{}'.format(zone_number, zone_letter),]

    np2ply(absolute[:, 0:3], os.path.join(work_dir, 'register/registered_sparse_points.ply'), comments)
    plot_reproj_err(absolute[:, -1], os.path.join(work_dir, 'register/inspect_rpc_reproj_err.jpg'))


if __name__ == '__main__':
    # work_dirs = ['/data2/kz298/core3d_result/aoi-d1-wpafb/',
    #              '/data2/kz298/core3d_result/aoi-d2-wpafb/',
    #              '/data2/kz298/core3d_result/aoi-d3-ucsd/',
    #              '/data2/kz298/core3d_result/aoi-d4-jacksonville/']

    work_dirs = ['/data2/kz298/mvs3dm_result/Explorer',
                '/data2/kz298/mvs3dm_result/MasterProvisional1',
                '/data2/kz298/mvs3dm_result/MasterProvisional2',
                '/data2/kz298/mvs3dm_result/MasterProvisional3',
                '/data2/kz298/mvs3dm_result/MasterSequestered1',
                '/data2/kz298/mvs3dm_result/MasterSequestered2',
                '/data2/kz298/mvs3dm_result/MasterSequestered3',
                '/data2/kz298/mvs3dm_result/MasterSequesteredPark']
    # colmap_dirs = [os.path.join(work_dir, 'colmap') for work_dir in work_dirs]

    import sys
    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    triangualte_all_points(work_dirs[0])