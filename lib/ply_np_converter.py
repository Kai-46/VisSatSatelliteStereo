#  ===============================================================================================================
#  Copyright (c) 2019, Cornell University. All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without modification, are permitted provided that
#  the following conditions are met:
#
#      * Redistributions of source code must retain the above copyright otice, this list of conditions and
#        the following disclaimer.
#
#      * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and
#        the following disclaimer in the documentation and/or other materials provided with the distribution.
#
#      * Neither the name of Cornell University nor the names of its contributors may be used to endorse or
#        promote products derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
#  WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
#  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE
#  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
#  TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
#  HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
#   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
#  OF SUCH DAMAGE.
#
#  Author: Kai Zhang (kz298@cornell.edu)
#
#  The research is based upon work supported by the Office of the Director of National Intelligence (ODNI),
#  Intelligence Advanced Research Projects Activity (IARPA), via DOI/IBC Contract Number D17PC00287.
#  The U.S. Government is authorized to reproduce and distribute copies of this work for Governmental purposes.
#  ===============================================================================================================


from lib.plyfile import PlyData, PlyElement
import numpy as np


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
