from rpc_triangulate_solver.triangulate_task import triangulate_task
import os
import json
from lib.rpc_model import RPCModel
import numpy as np
import logging


def triangulate_worker(result_file, meta_file, affine_file, track_file, tmp_file):
    # load RPC model and their affine approximation
    with open(affine_file) as fp:
        affine_dict = json.load(fp)

    with open(meta_file) as fp:
        metas_dict = json.load(fp)

    rpc_dict = {}
    for img_name in affine_dict:
        # projection matrix
        affine_dict[img_name] = np.array(affine_dict[img_name][2:]).reshape((2, 4))
        rpc_dict[img_name] = RPCModel(metas_dict[img_name])

    points = []
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

        final_point, err = triangulate_task(track, rpc_models, affine_models, tmp_file)

        points.append([final_point[0], final_point[1], final_point[2], err])

    # remove tmpfile.txt
    os.remove(tmp_file)

    # write to result_file
    np.save(result_file, points)

    logging.info('process {} done, triangulated {} points'.format(pid, cnt))
