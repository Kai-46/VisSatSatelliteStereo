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
