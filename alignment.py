import analyze.InspectSparseModel as InspectSparseModel
import rpc_triangulate.triangulate as triangulate

import os
import shutil
import json
from lib.rpc_model import RPCModel


def create_workspace(proj_dir):
    work_dir = os.path.join(proj_dir, 'align')
    if not os.path.exists(work_dir):
        os.mkdir(work_dir)

    sparse_dir = os.path.join(proj_dir, 'sparse_ba')
    sparse_inspector = InspectSparseModel(sparse_dir, work_dir)
    sparse_inspector.inspect()

    # copy skew
    if not os.path.exists(os.path.join(work_dir, 'skews.json')):
        shutil.copy(os.path.join(proj_dir, 'skews.json'), work_dir)

    # copy metas
    if not os.path.exists(os.path.join(work_dir, 'metas')):
        shutil.copytree(os.path.join(proj_dir, 'metas'), os.path.join(work_dir, 'metas'))

    # map feature track into grid before skew correction
    with open(os.path.join(work_dir, 'inspect_points_track.json')) as fp:
        tracks = json.load(fp)

    with open(os.path.join(work_dir, 'skews.json')) as fp:
        skews = json.load(fp)


    for i in range(len(tracks)):
        for j in range(len(tracks[i])):
            img_name, col, row = tracks[i][j]
            # add skew back
            col += skews[img_name] * row

            tracks[i][j] = (img_name, col, row)

    with open(os.path.join(work_dir, 'track_original.json'), 'w') as fp:
        json.dump(tracks, fp)

def triangulate_all_points(work_dir):
    with open(os.path.join(work_dir, 'track_original.json'), 'w') as fp:
        tracks = json.load(fp)

    img_2_rpc= {}
    for item in os.listdir(os.path.join(work_dir, 'metas')):
        # .json
        img_name = item[:-5] + '.png'
        with open(os.path.join(work_dir, 'metas/{}'.format(item))) as fp:
            img_2_rpc[img_name] = RPCModel(json.load(fp))

    with open(os.path.join(work_dir, 'roi.json')) as fp:
        roi_dict = json.load(fp)

    points = []
    for i in range(len(tracks)):
        track = []
        rpc_models = []
        for j in range(len(tracks[i])):
            img_name, col, row = tracks[i][j]
            # add skew back
            track.append((col, row))
            rpc_models.append(img_2_rpc[img_name])
        # create task
        out_file = os.path.join(work_dir, 'task.txt')
        point = triangulate(track, rpc_models, roi_dict, out_file)
        points.append(point)

    with open(os.path.join(work_dir, 'points.json'), 'w') as fp:
        json.dump(points, fp)