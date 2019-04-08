from lib.rpc_triangulate import triangulate
import os
import json
from lib.rpc_model import RPCModel
import numpy as np
import logging
import multiprocessing


def run_triangulation(result_file, work_dir, track_file, tmp_file):
    # load RPC model and affine approximation
    with open(os.path.join(work_dir, 'approx_affine_latlon.json')) as fp:
        affine_dict = json.load(fp)

    rpc_dict = {}
    for img_name in affine_dict:
        # projection matrix
        affine_dict[img_name] = np.array(affine_dict[img_name][2:]).reshape((2, 4))

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


def triangualte_all_points(work_dir, track_file, out_file, tmp_dir):
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
        track_file = os.path.join(tmp_dir, '{}_tracks_{}.txt'.format(pid, i))
        track_file_list.append(track_file)
        with open(track_file, 'w') as fp:
            idx1 = chunks[i][0]
            idx2 = chunks[i][1]
            json.dump(all_tracks[idx1:idx2], fp)
        tmp_file = os.path.join(tmp_dir, '{}_tmpfile_{}.txt'.format(pid, i))
        result_file = os.path.join(tmp_dir, '{}_absolute_{}.npy'.format(pid, i))
        result_file_list.append(result_file)

        p = multiprocessing.Process(target=run_triangulation, args=(result_file, work_dir, track_file, tmp_file))
        process_list.append(p)
        p.start()

    for p in process_list:
        p.join()

    # read result_files
    absolute = []
    for result_file in result_file_list:
        absolute.append(np.load(result_file))
    absolute = np.vstack(tuple(absolute))

    np.savetxt(out_file, absolute,
               header='# format: easting, northing, height, zone_number, zone_letter (1 for N and -1 for S), reproj_err')

    # remove all track_file
    for track_file in track_file_list:
        os.remove(track_file)
    for result_file in result_file_list:
        os.remove(result_file)


if __name__ == '__main__':
    pass
