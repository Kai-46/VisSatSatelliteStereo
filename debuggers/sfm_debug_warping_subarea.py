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
import shutil
import cv2
from pyquaternion import Quaternion
import multiprocessing


# compute the homography directly from the 3*4 projection matrix
# plane_vec is a 4 by 1 vector
# return homography matrix that warps the src image to the reference image
def compute_homography(ref_P, src_P, plane_vec):
    plane_normal = plane_vec[:3, :]
    plane_constant = plane_vec[3, 0]

    # first rewrite x=P[X; 1] as x=SX
    ref_S = (ref_P[:3, :3] + np.dot(ref_P[:3, 3:4], plane_normal.T) / plane_constant);
    src_S = (src_P[:3, :3] + np.dot(src_P[:3, 3:4], plane_normal.T) / plane_constant);

    # H = np.dot(src_S, np.linalg.inv(ref_S))
    H = np.dot(ref_S, np.linalg.inv(src_S))

    H = H / np.max(np.abs(H))   # increase numeric stability
    return H


def create_warped_images_worker(sweep_plane, camera_mat_dict, image_dir, ref_img_name, src_img_names, out_subdir_dict, avg_img_out_dir, subarea=None):
    i, plane_vec = sweep_plane
    ref_im = np.float32(cv2.imread(os.path.join(image_dir, ref_img_name))) / 255.0

    ul_x = 0
    ul_y = 0
    h, w = ref_im.shape[:2]
    if subarea is not None:
        ul_x, ul_y, w, h = subarea
        # make sure the subarea lies in the image
        assert (0 <= ul_x < ref_im.shape[1] - w and 0 <= ul_y < ref_im.shape[0] - h)
        ref_im = ref_im[ul_y:ul_y+h, ul_x:ul_x+w]

    print('ul_x, ul_y, w, h: {}, {}, {}, {}'.format(ul_x, ul_y, w, h ))

    cnt = 1
    for img_name in src_img_names:
        print('plane {}, ({},{},{},{}) warping {} to {}'.format(i, plane_vec[0, 0], plane_vec[1, 0], plane_vec[2, 0], plane_vec[3, 0],
                                                                        img_name, ref_img_name))
        # compute homography
        H = compute_homography(camera_mat_dict[ref_img_name], camera_mat_dict[img_name], plane_vec)
        # debug
        # print('H: {}'.format(H))

        # warp image
        im = np.float32(cv2.imread(os.path.join(image_dir, img_name))) / 255.0
        
        # modify H mat
        translation = np.array([[1.0, 0.0, -ul_x], 
                                [0.0, 1.0, -ul_y], 
                                [0.0, 0.0, 1.0]])
        H = np.dot(translation, H)

        warped_im = cv2.warpPerspective(im, H, (w, h), borderMode=cv2.BORDER_CONSTANT)
        cv2.imwrite(os.path.join(out_subdir_dict[img_name], 'warped_plane_{:04}.jpg'.format(i)), np.uint8(warped_im * 255.0))

        ref_im += warped_im
        cnt += 1
    ref_im /= cnt
    cv2.imwrite(os.path.join(avg_img_out_dir, 'avg_plane_{:04}.jpg'.format(i)), np.uint8(ref_im * 255.0))


def create_warped_images(sfm_perspective_dir, ref_img_id, z_min, z_max, num_planes, normal, out_dir, src_img_ids=[], max_processes=None, subarea=None):
    # for each source image, it would create a folder to save the warped images for all the sweeping planes
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.mkdir(out_dir)

    # load camera matrices
    # with open(os.path.join(sfm_perspective_dir, 'init_camera_dict.json')) as fp:
    with open(os.path.join(sfm_perspective_dir, 'init_ba_camera_dict.json')) as fp:
        camera_dict = json.load(fp)

    if len(src_img_ids) == 0:
        useful_img_ids = []
        for item in camera_dict.keys():
            idx = item.find('_')
            id = int(item[:idx])
            useful_img_ids.append(id)
    else:
        if ref_img_id in src_img_ids:
            raise Exception('ref_img_id should not appear in src_img_ids')
        else:
            useful_img_ids = [ref_img_id, ] + src_img_ids

    img_id2name = {}
    img_name2id = {}
    camera_size_dict = {}
    camera_mat_dict = {}
    for item in camera_dict.keys():
        idx = item.find('_')
        id = int(item[:idx])
        if id not in useful_img_ids:
            continue

        img_id2name[id] = item
        img_name2id[item] = id

        params = camera_dict[item]
        w, h, fx, fy, cx, cy, s = params[:7]
        camera_size_dict[item] = (w, h)

        qvec = params[7:11]
        tvec = params[11:14]
        K = np.array([[fx, s, cx],
                      [0.0, fy, cy],
                      [0.0, 0.0, 1.0]])
        R = Quaternion(qvec[0], qvec[1], qvec[2], qvec[3]).rotation_matrix
        t = np.array(tvec).reshape((3, 1))
        P = np.dot(K, np.hstack((R, t)))
        P = P / np.max(np.abs(P))   # increase numeric stability
        camera_mat_dict[item] = P

    ref_img_name = img_id2name[ref_img_id]

    out_subdir_dict = {}
    src_img_names = []
    for item in sorted(img_name2id.keys()):
        if img_name2id[item] == ref_img_id:
            continue
        src_img_names.append(item)
        # create output folders; one for each source image
        subdir = os.path.join(out_dir, item[:-4])   # remove '.png' suffix
        if os.path.exists(subdir):  # remove existing subdir
            shutil.rmtree(subdir)
        os.mkdir(subdir)
        out_subdir_dict[item] = subdir

    # loop over all the sweep planes
    z_sequence = np.linspace(z_min, z_max, num_planes)
    sweep_plane_sequence = []
    for i in range(len(z_sequence)):
        plane_vec = np.array([normal[0], normal[1], normal[2], z_sequence[i]]).reshape(4, 1) 
        sweep_plane_sequence.append((i, plane_vec))

    image_dir = os.path.join(sfm_perspective_dir, 'images')
    print('Start warping all the images to the same reference view: {}...'.format(ref_img_name))
    # shutil.copy2(os.path.join(image_dir, ref_img_name), out_dir)
    ref_im = cv2.imread(os.path.join(image_dir, ref_img_name))
    if subarea is not None:
        ul_x, ul_y, w, h = subarea
        # make sure the subarea lies in the image
        assert (0 <= ul_x < ref_im.shape[1] - w and 0 <= ul_y < ref_im.shape[0] - h)
        ref_im = ref_im[ul_y:ul_y+h, ul_x:ul_x+w]
    cv2.imwrite(os.path.join(out_dir, ref_img_name[:-4]+'.jpg'), ref_im)
    
    avg_img_out_dir = os.path.join(out_dir, 'avg_img')
    if os.path.exists(avg_img_out_dir):
        shutil.rmtree(avg_img_out_dir)
    os.mkdir(avg_img_out_dir)

    if max_processes is None:
        max_processes = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(max_processes)

    results = []
    for sweep_plane in sweep_plane_sequence:
        r = pool.apply_async(create_warped_images_worker, 
                            (sweep_plane, camera_mat_dict, image_dir, ref_img_name, src_img_names, out_subdir_dict, avg_img_out_dir, subarea))
        results.append(r)

    [r.wait() for r in results]     # sync

    # create an average video
    cmd = 'ffmpeg -y -framerate 25 -i  {} \
            -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" \
            {}'.format(os.path.join(avg_img_out_dir, 'avg_plane_%04d.jpg'),
                       os.path.join(out_dir, 'avg_img.mp4'))
    os.system(cmd)


def debug():
    # warp the other images to this reference image frame
    ref_img_id = 5
    # src_img_ids = [6, 7]
    src_img_ids = []      # if set to empty list, then all the other images except the reference image will be warped

    # mvs3dm explorer
    sfm_perspective_dir = '/data2/kz298/core3d_result/explorer/colmap/sfm_perspective'
    out_dir = '/data2/kz298/core3d_result/explorer/debug_warping'
    # height range
    z_min = -30  # meters
    z_max = 120
    # number of sweeping planes and normal direction
    # a plane with height z is written as n^x-z=0
    num_planes = 500
    # num_planes = 80
    normal = (0, 0, 1)
    subarea = (2573, 1449, 128, 128) 

    max_processes = 10
    create_warped_images(sfm_perspective_dir, ref_img_id, z_min, z_max, num_planes, normal, out_dir, src_img_ids, max_processes, subarea)


if __name__ == '__main__':
    debug()
