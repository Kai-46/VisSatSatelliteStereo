from lib.plyfile import PlyData, PlyElement
import numpy as np


def np2ply(data, out_ply, comments=None):
    dim = data.shape[1]
    assert (dim in [3, 6, 9])

    data = list(zip(*[data[:, i] for i in range(dim)]))
    if dim == 3:
        vertex = np.array(data, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    elif dim == 6:
        vertex = np.array(data, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                                    ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4')])
    else:
        vertex = np.array(data, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                                    ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
                                    ('red', 'uint8'), ('green', 'uint8'), ('blue', 'uint8')])

    el = PlyElement.describe(vertex, 'vertex')
    if comments is None:
        PlyData([el], byte_order='<').write(out_ply)
    else:
        PlyData([el], byte_order='<', comments=comments).write(out_ply)


def ply2np(in_ply):
    vertex = PlyData.read(in_ply)['vertex'].data

    names = vertex.dtype.names

    data = []
    if 'x' in names:
        data.append(np.hstack((vertex['x'].reshape((-1, 1)),
                              vertex['y'].reshape((-1, 1)),
                              vertex['z'].reshape((-1, 1)))))
    if 'nx' in names:
        data.append(np.hstack((vertex['nx'].reshape((-1, 1)),
                              vertex['ny'].reshape((-1, 1)),
                              vertex['nz'].reshape((-1, 1)))))
    if 'red' in names:
        data.append(np.hstack((vertex['red'].reshape((-1, 1)),
                              vertex['green'].reshape((-1, 1)),
                              vertex['blue'].reshape((-1, 1)))))

    data = np.hstack(tuple(data))
    return data


if __name__ == '__main__':
    data = np.random.randn(500, 9)

    np2ply(data, '/data2/tmp.ply')

    ply2np('/data2/tmp.ply')