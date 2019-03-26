import json
import utm
from lib.rpc_model import RPCModel
import numpy as np
from lib.run_cmd import run_cmd
import logging
import os


def compute_reproj_err(rpc_models, track, point):
    assert (len(rpc_models) == len(track))

    err = 0.

    lat, lon, alt = point

    cnt = len(track)
    for i in range(len(track)):
        col, row = track[i]
        proj_col, proj_row = rpc_models[i].projection(lat, lon, alt)

        err += np.sqrt((col - proj_col) ** 2 + (row - proj_row) ** 2)
    err = err / cnt

    # logging.info('reproj. err.: {}'.format(err))
    return err


# affine_model is an approximation of the rpc model
def solve_init(track, rpc_models, affine_models):
    cnt = len(track)
    assert(len(rpc_models) == cnt and len(affine_models) == cnt)

    A = np.zeros((2 * cnt, 3))
    b = np.zeros((2 * cnt, 1))

    for i in range(cnt):
        A[i, :] = affine_models[i][0, 0:3]
        b[i, 0] = track[i][0] - affine_models[i][0, 3]

        A[cnt + i, :] = affine_models[i][1, 0:3]
        b[cnt + i, 0] = track[i][1] - affine_models[i][1, 3]

    res = np.linalg.lstsq(A, b, rcond=-1)
    init = res[0].reshape((3, ))

    return init


def write_to_taskfile(track, rpc_models, init, out_file):
    assert (len(rpc_models) == len(track))

    lines = []
    lines.append('{} {} {}\n'.format(init[0], init[1], init[2]))

    # err = compute_reproj_err(rpc_models, track, init)
    # lines.append('{}\n'.format(err))

    for i in range(len(track)):
        line = '{} {}'.format(track[i][0], track[i][1])

        assert (len(rpc_models[i].rowNum) == 20)
        for j in range(20):
            line += ' {}'.format(rpc_models[i].rowNum[j])
        for j in range(20):
            line += ' {}'.format(rpc_models[i].rowDen[j])
        for j in range(20):
            line += ' {}'.format(rpc_models[i].colNum[j])
        for j in range(20):
            line += ' {}'.format(rpc_models[i].colDen[j])
        line += ' {} {} {} {} {} {} {} {} {} {}\n'.format(rpc_models[i].latOff, rpc_models[i].latScale,
                                                        rpc_models[i].lonOff, rpc_models[i].lonScale,
                                                        rpc_models[i].altOff, rpc_models[i].altScale,
                                                        rpc_models[i].rowOff, rpc_models[i].rowScale,
                                                        rpc_models[i].colOff, rpc_models[i].colScale)
        lines.append(line)
    with open(out_file, 'w') as fp:
        fp.writelines(lines)


# track is a list of observations
def triangulate(track, rpc_models, affine_models, out_file):
    init = solve_init(track, rpc_models, affine_models)
    write_to_taskfile(track, rpc_models, init, out_file)

    # triangulate
    tmp_file = '{}.result.txt'.format(out_file)
    run_cmd('/data2/kz298/satellite_stereo/multi_rpc_triangule/multi_rpc_triangulate {} {}'.format(out_file, tmp_file), disable_log=True)

    #run_cmd('/home/kai/satellite_stereo/satellite_stereo/multi_rpc_triangule/multi_rpc_triangulate {} {}'.format(
    #    out_file, tmp_file), disable_log=True)
    # read result
    with open(out_file) as fp:
        lines = fp.readlines()
        init_point = [float(x) for x in lines[0].strip().split(' ')]
        # init_err = float(lines[1].strip())
    with open(tmp_file) as fp:
        lines = fp.readlines()
        final_point = [float(x) for x in lines[0].strip().split(' ')]
        # final_err = float(lines[1].strip())

    # remove tmpfile
    os.remove(tmp_file)

    # double check the final_err
    # init_err = compute_reproj_err(rpc_models, track, init)
    final_err = compute_reproj_err(rpc_models, track, final_point)
    # #print('here ceres final_err: {}, python: {}'.format(final_err, tmp))
    # assert (np.abs(init_err - final_err) < 0.001)

    # tmp = utm.from_latlon(init_point[0], init_point[1])
    # init_point_utm = [tmp[0], tmp[1], init_point[2], tmp[2], tmp[3]]

    tmp = utm.from_latlon(final_point[0], final_point[1])
    final_point_utm = [tmp[0], tmp[1], final_point[2], tmp[2], tmp[3]]

    # logging.info('init point: lat,lon,alt={}, utm_east,utm_north,alt,zone_number,zone_letter={}, reproj. err.: {}'.format(init_point, init_point_utm, init_err))
    # logging.info('final point: lat,lon,alt={}, utm_east,utm_north,alt,zone_number,zone_letter={}, reproj. err.: {}'.format(final_point, final_point_utm, final_err))

    return final_point, final_point_utm, final_err


def test():
    rpc_models = []

    rpc_dict_files = ['example_triangulate/000_20141005160138.json',
                      'example_triangulate/001_20141005160149.json',
                      'example_triangulate/002_20141011155720.json']
    for fname in rpc_dict_files:
        with open(fname) as fp:
            meta_dict = json.load(fp)
            rpc_models.append(RPCModel(meta_dict))

    track = []
    with open('example_triangulate/track.txt') as fp:
        for line in fp.readlines():
            tmp = line.strip().split(' ')
            track.append((float(tmp[0]), float(tmp[1])))


    with open('example_triangulate/roi.json') as fp:
        roi_dict = json.load(fp)

    ul_east = roi_dict['x']
    ul_north = roi_dict['y']
    lr_east = ul_east + roi_dict['w']
    lr_north = ul_north - roi_dict['h']
    zone_letter = roi_dict['zone_letter']
    zone_number = roi_dict['zone_number']

    init = solve_init(rpc_models, track, zone_number, zone_letter, ul_east, ul_north, lr_east, lr_north)
    write_to_taskfile(init, rpc_models, track, 'example_triangulate/task.txt')


if __name__ == '__main__':
    test()