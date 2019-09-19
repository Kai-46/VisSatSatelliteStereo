# ===============================================================================================================
#  This file is part of Creation of Operationally Realistic 3D Environment (CORE3D).                            =
#  Copyright 2019 Cornell University - All Rights Reserved                                                      =
#  -                                                                                                            =
#  NOTICE: All information contained herein is, and remains the property of General Electric Company            =
#  and its suppliers, if any. The intellectual and technical concepts contained herein are proprietary          =
#  to General Electric Company and its suppliers and may be covered by U.S. and Foreign Patents, patents        =
#  in process, and are protected by trade secret or copyright law. Dissemination of this information or         =
#  reproduction of this material is strictly forbidden unless prior written permission is obtained              =
#  from General Electric Company.                                                                               =
#  -                                                                                                            =
#  The research is based upon work supported by the Office of the Director of National Intelligence (ODNI),     =
#  Intelligence Advanced Research Projects Activity (IARPA), via DOI/IBC Contract Number D17PC00287.            =
#  The U.S. Government is authorized to reproduce and distribute copies of this work for Governmental purposes. =
# ===============================================================================================================

from lib.plyfile import PlyData, PlyElement
import numpy as np

# def np2ply(data, out_ply, comments=None, text=False):
#     dim = data.shape[1]
#     assert (dim in [3, 6, 9])

#     data = list(zip(*[data[:, i] for i in range(dim)]))
#     if dim == 3:
#         vertex = np.array(data, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
#     elif dim == 6:
#         vertex = np.array(data, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
#                                     ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4')])
#     else:
#         vertex = np.array(data, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
#                                     ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
#                                     ('red', 'uint8'), ('green', 'uint8'), ('blue', 'uint8')])

#     el = PlyElement.describe(vertex, 'vertex')
#     if text:
#         if comments is None:
#             PlyData([el], text=True).write(out_ply)
#         else:
#             PlyData([el], text=True, comments=comments).write(out_ply)
#     else:
#         if comments is None:
#             PlyData([el], byte_order='<').write(out_ply)
#         else:
#             PlyData([el], byte_order='<', comments=comments).write(out_ply)

# only support writing vertex and color attributes
def np2ply(vertex, out_ply, color=None, comments=None, text=False, use_double=False):
    if use_double:
        vertex = vertex.astype(dtype=np.float64)
        dtype_list = [('x', 'f8'), ('y', 'f8'), ('z', 'f8')]
    else:
        vertex = vertex.astype(dtype=np.float32)
        dtype_list = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')] 

    if color is None:
        data = [(vertex[i, 0], vertex[i, 1], vertex[i, 2]) for i in range(vertex.shape[0])]
    else:
        data = [(vertex[i, 0], vertex[i, 1], vertex[i, 2], color[i, 0], color[i, 1], color[i, 2]) for i in range(vertex.shape[0])]
        dtype_list = dtype_list + [('red', 'uint8'), ('green', 'uint8'), ('blue', 'uint8')]

    vertex = np.array(data, dtype=dtype_list)
    el = PlyElement.describe(vertex, 'vertex')
    if text:
        text_fmt = ['%.4f', '%.4f', '%.4f']
        if color is not None:
            text_fmt = text_fmt + ['%i', '%i', '%i']

        if comments is None:
            PlyData([el], text=True, text_fmt=text_fmt).write(out_ply)
        else:
            PlyData([el], text=True, text_fmt=text_fmt, comments=comments).write(out_ply)
    else:
        if comments is None:
            PlyData([el], byte_order='<').write(out_ply)
        else:
            PlyData([el], byte_order='<', comments=comments).write(out_ply)

# def ply2np(in_ply, return_comments=False, only_xyz=True):
#     ply = PlyData.read(in_ply)
#     comments = ply.comments

#     vertex = ply['vertex'].data
#     names = vertex.dtype.names

#     data = []
#     if 'x' in names:
#         data.append(np.hstack((vertex['x'].reshape((-1, 1)),
#                               vertex['y'].reshape((-1, 1)),
#                               vertex['z'].reshape((-1, 1)))))
#     if not only_xyz:
#         if 'nx' in names:
#             data.append(np.hstack((vertex['nx'].reshape((-1, 1)),
#                                   vertex['ny'].reshape((-1, 1)),
#                                   vertex['nz'].reshape((-1, 1)))))
#         if 'red' in names:
#             data.append(np.hstack((vertex['red'].reshape((-1, 1)),
#                                   vertex['green'].reshape((-1, 1)),
#                                   vertex['blue'].reshape((-1, 1)))))

#     data = np.hstack(tuple(data))
#     if return_comments:
#         return data, comments
#     else:
#         return data

# not support surface normal
def ply2np(in_ply):
    ply = PlyData.read(in_ply)
    comments = ply.comments
    if len(comments) == 0:
        comments = None
    
    vertex = ply['vertex'].data
    names = vertex.dtype.names

    if 'x' in names:
        data = np.hstack((vertex['x'].reshape((-1, 1)),
                          vertex['y'].reshape((-1, 1)),
                          vertex['z'].reshape((-1, 1))))

    if 'red' in names:
        color = np.hstack((vertex['red'].reshape((-1, 1)),
                           vertex['green'].reshape((-1, 1)),
                           vertex['blue'].reshape((-1, 1))))
    else:
        color = None

    return data, color, comments


if __name__ == '__main__':
    data = np.random.randn(500, 9)

    np2ply(data, '/data2/tmp.ply')

    ply2np('/data2/tmp.ply')
