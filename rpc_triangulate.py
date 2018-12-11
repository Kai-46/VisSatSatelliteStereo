import json
import utm
from lib.rpc_model import RPCModel
import numpy as np
from lib.gen_grid import gen_grid

def solve_affine(xx, yy, zz, col, row):
    # construct a least square problem
    print('xx: {}, {}'.format(np.min(xx), np.max(xx)))
    print('yy: {}, {}'.format(np.min(yy), np.max(yy)))
    print('zz: {}, {}'.format(np.min(zz), np.max(zz)))
    print('col: {}, {}'.format(np.min(col), np.max(col)))
    print('row: {}, {}'.format(np.min(row), np.max(row)))

    diff_size = np.array([yy.size - xx.size, zz.size - xx.size, col.size - xx.size, row.size - xx.size])
    assert (np.all(diff_size == 0))

    point_cnt = xx.size
    all_ones = np.ones((point_cnt, 1))
    all_zeros = np.zeros((point_cnt, 4))
    # construct the least square problem
    A1 = np.hstack((xx, yy, zz, all_ones, all_zeros))
    A2 = np.hstack((all_zeros, xx, yy, zz, all_ones))

    A = np.vstack((A1, A2))
    b = np.vstack((col, row))
    res = np.linalg.lstsq(A, b)

    print('residual error (pixels): {}'.format(np.sqrt(res[1][0] / point_cnt)))

    P = res[0].reshape((2, 4))

    return P


def approx_affine(rpc_model, ul_lat, ul_lon, lr_lat, lr_lon):
    z_min = rpc_model.altOff - 0. * rpc_model.altScale
    z_max = rpc_model.altOff + 0.7 * rpc_model.altScale

    xx, yy, zz = gen_grid(ul_lat, lr_lat, ul_lon, lr_lon, z_min, z_max)
    col, row = rpc_model.projection(xx, yy, zz)

    P = solve_affine(xx, yy, zz, col, row)

    return P


def compute_reproj_err(rpc_models, track, point):
    assert (len(rpc_models) == len(track))

    err = 0.

    lat, lon, alt = point

    for i in range(len(track)):
        col, row = track[i]
        proj_col, proj_row = rpc_models[i].projection(lat, lon, alt)

        err += (col - proj_col) ** 2 + (row - proj_row) ** 2
    err = np.sqrt(err / len(track))

    return err


def solve_init(rpc_models, track, zone_number, zone_letter, ul_east, ul_north, lr_east, lr_north):
    assert (len(rpc_models) == len(track))

    ul_lat, ul_lon = utm.to_latlon(ul_east, ul_north, zone_number, zone_letter)
    lr_lat, lr_lon = utm.to_latlon(lr_east, lr_north, zone_number, zone_letter)

    P = []
    cnt = len(track)
    for i in range(cnt):
        tmp = approx_affine(rpc_models[i], ul_lat, ul_lon, lr_lat, lr_lon)
        P.append(tmp)

    A = np.zeros((2 * cnt, 3))
    b = np.zeros((2 * cnt, 1))

    for i in range(cnt):
        A[i, :] = P[i][0, 0:3]
        b[i, 0] = track[i][0] - P[i][0, 3]

        A[cnt + i, :] = P[i][1, 0:3]
        b[cnt + i, 0] = track[i][1] - P[i][1, 3]

    res = np.linalg.lstsq(A, b)
    init = res[0].reshape((3, ))

    return init


def write_to_taskfile(init, rpc_models, track, out_file):
    assert (len(rpc_models) == len(track))

    lines = []
    lines.append('{} {} {}\n'.format(init[0], init[1], init[2]))

    err = compute_reproj_err(rpc_models, track, init)
    lines.append('{}\n'.format(err))

    for i in range(len(track)):
        line = '{} {}'.format(track[0], track[1])

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


def test():
    rpc_models = []

    rpc_dict_files = ['example_triangulate/000_20141005160138.json',
                      'example_triangulate/001_20141005160149.json',
                      'example_triangulate/002_20141011155720.json']
    for fname in rpc_dict_files:
        with open(fname) as fp:
            rpc_dict = json.load(fp)['rpc']
            rpc_models.append(RPCModel(rpc_dict))

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