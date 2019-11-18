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


import os
import json
import numpy as np

import sys
# add satellite_stereo to python search path
sys.path.append(os.path.join("/data2/kz298/satellite_stereo"))

# {work_dir}/colmap/sfm_pinhole
sfm_pinhole_dir = '/data2/kz298/core3d_result/aoi-d4-jacksonville/colmap/sfm_pinhole'

out_dir = os.path.join(sfm_pinhole_dir, 'debug')

import shutil
if os.path.exists(out_dir):
    shutil.rmtree(out_dir)
os.mkdir(out_dir)

with open(os.path.join(sfm_pinhole_dir, 'init_camera_dict.json')) as fp:
    camera_dict = json.load(fp)

img_id2name = {}
for item in camera_dict.keys():
    idx = item.find('_')
    id = int(item[:idx])
    img_id2name[id] = item


from colmap.extract_sfm import extract_all_to_dir
extract_all_to_dir(os.path.join(sfm_pinhole_dir, 'init_triangulate'), out_dir)


for img_id in sorted(img_id2name.keys()):
    print('processing {}...'.format(img_id2name[img_id]))

    params = camera_dict[img_id2name[img_id]]
    w, h, fx, fy, cx, cy = params[:6]
    qvec = params[6:10]
    tvec = params[10:13]

    K = np.array([[fx, 0.0, cx],
                  [0.0, fy, cy],
                  [0.0, 0.0, 1.0]])
    from pyquaternion import Quaternion
    R = Quaternion(qvec[0], qvec[1], qvec[2], qvec[3]).rotation_matrix
    t = np.array(tvec).reshape((3, 1))

    import imageio
    base_image = imageio.imread(os.path.join(sfm_pinhole_dir, 'images', img_id2name[img_id]))
    base_image = np.tile(base_image[:, :, np.newaxis], (1, 1, 3))

    points = np.loadtxt(os.path.join(out_dir, 'kai_coordinates.txt'))

    for i in range(points.shape[0]):
        x = points[i, :3].reshape((3, 1))
        tmp = np.dot(K, np.dot(R, x) + t)
        c = int(tmp[0, 0] / tmp[2, 0])
        r = int(tmp[1, 0] / tmp[2, 0])
        circle_size = 5
        if circle_size <= c < base_image.shape[1]-circle_size and circle_size <= r < base_image.shape[0]-circle_size:
            base_image[r-circle_size:r+circle_size, c-circle_size:c+circle_size, :] = np.array([0, 255, 0])

    imageio.imwrite(os.path.join(out_dir, img_id2name[img_id][:-4]+'.jpg'), base_image)
