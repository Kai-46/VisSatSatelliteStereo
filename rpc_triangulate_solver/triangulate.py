from rpc_triangulate_solver.triangulate_worker import triangulate_worker
import os
import json
import numpy as np
import logging
import multiprocessing
import shutil


def triangulate(meta_file, affine_file, track_file, out_file, tmp_dir):
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)

    with open(track_file) as fp:
        all_tracks = json.load(fp)

    pid = os.getpid()

    # split all_tracks into multiple chunks
    process_cnt = multiprocessing.cpu_count()
    process_list = []

    chunk_size = int(len(all_tracks) / process_cnt)
    chunks = [[j*chunk_size, (j+1)*chunk_size] for j in range(process_cnt)]
    chunks[-1][1] = len(all_tracks)
    track_file_list = []
    result_file_list = []
    for i in range(process_cnt):
        track_file = os.path.join(tmp_dir, 'rpc_triangulate_tracks_{}_{}.txt'.format(pid, i))
        track_file_list.append(track_file)
        with open(track_file, 'w') as fp:
            idx1 = chunks[i][0]
            idx2 = chunks[i][1]
            json.dump(all_tracks[idx1:idx2], fp)
        tmp_file = os.path.join(tmp_dir, 'rpc_triangulate_tmpfile_{}_{}.txt'.format(pid, i))
        result_file = os.path.join(tmp_dir, 'rpc_triangulate_points_{}_{}.npy'.format(pid, i))
        result_file_list.append(result_file)

        p = multiprocessing.Process(target=triangulate_worker, args=(result_file, meta_file, affine_file, track_file, tmp_file))
        process_list.append(p)
        p.start()

    for p in process_list:
        p.join()

    # read result_files
    points = []
    for result_file in result_file_list:
        points.append(np.load(result_file))
    points = np.vstack(tuple(points))

    np.savetxt(out_file, points, header='# format: lat, lon, alt, reproj_err')

    # remove all track_file
    shutil.rmtree(tmp_dir)


if __name__ == '__main__':
    from lib.logger import GlobalLogger
    logger = GlobalLogger()
    logger.turn_on_terminal()

    folder = '/data2/kz298/satellite_stereo/rpc_triangulate_solver/example/'
    meta_file = os.path.join(folder, 'metas.json')
    affine_file = os.path.join(folder, 'affine_latlonalt.json')
    track_file = os.path.join(folder, 'kai_tracks.json')
    out_file = os.path.join(folder, 'points.txt')
    tmp_dir = os.path.join(folder, 'tmp')
    triangulate(meta_file, affine_file, track_file, out_file, tmp_dir)
