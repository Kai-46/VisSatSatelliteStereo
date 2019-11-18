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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pyquaternion import Quaternion
from visualization.plot_reproj_err import plot_reproj_err
from lib.ply_np_converter import np2ply
import shutil
from colmap.extract_sfm import extract_all_to_dir
from colmap.extract_raw_matches import extract_raw_matches
import json
import imageio


def vector_angle(vec1, vec2):
    vec1 = vec1 / np.sqrt(np.dot(vec1.T, vec1))
    vec2 = vec2 / np.sqrt(np.dot(vec2.T, vec2))

    tmp = np.dot(vec1.T, vec2)
    angle = np.rad2deg(np.math.acos(tmp))

    return angle


class SparseInspector(object):
    def __init__(self, sparse_dir, db_path, out_dir, camera_model, ext='.txt'):
        assert (camera_model == 'PINHOLE' or camera_model == 'PERSPECTIVE')
        self.camera_model = camera_model
        self.out_dir = out_dir
        if not os.path.exists(self.out_dir):
            os.mkdir(self.out_dir)

        self.db_path = db_path

        # extract colmap sfm results
        extract_all_to_dir(sparse_dir, self.out_dir, ext)
        with open(os.path.join(self.out_dir, 'kai_cameras.json')) as fp:
            camera_params = json.load(fp)

        self.points = np.loadtxt(os.path.join(self.out_dir, 'kai_points.txt'))

        with open(os.path.join(self.out_dir, 'kai_tracks.json')) as fp:
            self.tracks = json.load(fp)

        with open(os.path.join(self.out_dir, 'kai_keypoints.json')) as fp:
            self.view_keypoints = json.load(fp)

        self.img_names = sorted(camera_params.keys())
        self.camera_mats = {}
        self.img_sizes = {}
        for img_name in self.img_names:
            params = camera_params[img_name]
            w, h = params[:2]
            self.img_sizes[img_name] = (w, h)
            if camera_model == 'PINHOLE':
                fx, fy, cx, cy = params[2:6]
                s = 0
                qvec = params[6:10]
                tvec = params[10:13]
            else:
                fx, fy, cx, cy, s = params[2:7]
                qvec = params[7:11]
                tvec = params[11:14]

            K = np.array([[fx, s, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
            R = Quaternion(qvec[0], qvec[1], qvec[2], qvec[3]).rotation_matrix
            t = np.array(tvec).reshape((3, 1))
            self.camera_mats[img_name] = (K, R, t)

        self.img_cnt = len(self.camera_mats.keys())

    def inspect_all(self):
        self.inspect_raw_matches()
        self.inspect_tracks()
        self.inspect_image_key_points()
        self.inspect_scene_points()
        self.inspect_angles()

    def inspect_raw_matches(self):
        raw_matches_cnt = extract_raw_matches(self.db_path)

        out_subdir = os.path.join(self.out_dir, 'raw_matches')
        if not os.path.exists(out_subdir):
            os.mkdir(out_subdir)

        for img_name1 in self.img_names:
            tmp_dict = raw_matches_cnt[img_name1]
            with open(os.path.join(out_subdir, img_name1[:-4]+'.txt'), 'w') as fp:
                fp.write(img_name1 + '\n')
                for img_name2 in self.img_names:
                    if img_name2 == img_name1:
                        continue

                    if img_name2 in tmp_dict:
                        fp.write('{}, {}\n'.format(img_name2, tmp_dict[img_name2]))
                    else:
                        fp.write('{}, 0\n'.format(img_name2))

    def inspect_image_key_points(self):
        # per-view re-projection errors and depth ranges
        view_reproj_errs = np.zeros((self.img_cnt, 2))
        view_track_lens = np.zeros((self.img_cnt, 2))
        view_depth_ranges = np.zeros((self.img_cnt, 2))
        used_keypoint_cnt = np.zeros((self.img_cnt, ), dtype=np.int64)
        locations = []
        for idx, img_name in enumerate(self.img_names):
            keypoints = self.view_keypoints[img_name]
            cnt = len(keypoints)
            used_keypoint_cnt[idx] = cnt

            reproj_errs = np.zeros((cnt, ))
            track_lens = np.zeros((cnt, ))
            depth_ranges = np.zeros((cnt, ))
            K, R, tvec = self.camera_mats[img_name]
            w, h = self.img_sizes[img_name]
            location_image = np.zeros((h, w), dtype=np.uint8)
            for key_point_idx, key_point in enumerate(keypoints):
                u, v, x, y, z, track_len = key_point
                xyz = np.array([x, y, z]).reshape((3, 1))
                xyz = np.dot(R, xyz) + tvec
                
                depth_ranges[key_point_idx] = xyz[2, 0]

                tmp = np.dot(K, xyz)
                u1 = tmp[0, 0] / tmp[2, 0]
                v1 = tmp[1, 0] / tmp[2, 0]
                err = np.sqrt((u - u1) ** 2 + (v - v1) ** 2)
                reproj_errs[key_point_idx] = err

                track_lens[key_point_idx] = track_len

                radius = 5
                col_idx1 = max([int(u) - radius, 0])
                col_idx2 = min([int(u) + radius, w-1])
                row_idx1 = max([int(v) - radius, 0])
                row_idx2 = min([int(v) + radius, h-1])
                location_image[row_idx1:row_idx2+1, col_idx1:col_idx2+1] = 255

            locations.append(location_image)
            view_reproj_errs[idx, :] = (np.mean(reproj_errs), np.median(reproj_errs))
            view_track_lens[idx, :] = (np.mean(track_lens), np.median(track_lens))
            view_depth_ranges[idx, :] = np.percentile(depth_ranges, (1, 99))

        # write to text
        with open(os.path.join(self.out_dir, 'sfm_keypoints.txt'), 'w') as fp:
            fp.write('# img_name, img_width, img_height, used_keypoint_cnt, mean_reproj_err (pixels), median_reproj_err (pixels), mean_track_len, median_track_len\n')
            for idx, img_name in enumerate(self.img_names):
                w, h = self.img_sizes[img_name]
                fp.write('{}, {}, {}, {}, {:.4f}, {:.4f}, {:.4f}, {:.4f}\n'.format(img_name, w, h,  
                    used_keypoint_cnt[idx], view_reproj_errs[idx, 0], view_reproj_errs[idx, 1],
                    view_track_lens[idx, 0], view_track_lens[idx, 1]))

        with open(os.path.join(self.out_dir, 'sfm_depth_ranges.txt'), 'w') as fp:
            fp.write('# img_name, depth_min (meters), depth_max (meters)\n')
            for idx, img_name in enumerate(self.img_names):
                fp.write('{}, {}, {}\n'.format(img_name, view_depth_ranges[idx, 0], view_depth_ranges[idx, 1]))

        # plot distributions of the key points in the image space
        out_subdir = os.path.join(self.out_dir, 'image_keypoints_locations')
        if not os.path.exists(out_subdir):
            os.mkdir(out_subdir)

        for idx, img_name in enumerate(self.img_names):
            imageio.imwrite(os.path.join(out_subdir, img_name[:-4]+'.jpg'), locations[idx])


    def inspect_tracks(self):
        reproj_errs = self.points[:, 3]
        track_len = self.points[:, 4]   # fourth column

        plot_reproj_err(reproj_errs, os.path.join(self.out_dir, 'sfm_track_reproj_err.jpg'))

        # plot distribution of track_len
        plt.figure(figsize=(14, 5), dpi=80)
        max_track_len = int(np.max(track_len))
        plt.hist(track_len, bins=np.arange(0.5, max_track_len + 1.5, 1))
        plt.xticks(range(1, max_track_len+1))
        plt.ylabel('# of tracks')
        plt.xlabel('track length')
        plt.title('total # of images: {}\ntotal # of tracks: {}\ntrack length, min: {}, mean: {:.6f}, median: {}, max: {}'
                  .format(self.img_cnt, len(track_len), np.min(track_len), np.mean(track_len), np.median(track_len), max_track_len))
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, 'sfm_track_len.jpg'))
        plt.close()

    def inspect_scene_points(self):
        np2ply(self.points[:, 0:3], os.path.join(self.out_dir, 'sfm_points.ply'), color=self.points[:, -3:])

    def inspect_angles(self):
        cam_center_positions = []
        img_center_rays = []

        img_leftright_angles = []
        img_updown_angles = []
        img_angle_variations = []

        # all_rotations = []
        # all_translations = []
        for img_name in self.img_names:
            width, height = self.img_sizes[img_name]
            K, R, t = self.camera_mats[img_name]

            cam_center_positions.append(np.dot(R.T, -t))

            p1 = np.dot(np.linalg.inv(K), np.array([width / 2., height / 2., 1.]).reshape(3, 1))
            p2 = np.dot(np.linalg.inv(K), np.array([width / 2., height / 2., 10000.]).reshape(3, 1))
            img_center_rays.append(np.dot(R.T, p1 - t) - np.dot(R.T, p2 - t))

            left = np.dot(np.linalg.inv(K), np.array([0., height / 2., 1.]).reshape(3, 1))
            right = np.dot(np.linalg.inv(K), np.array([width, height / 2., 1.]).reshape(3, 1))
            img_leftright_angles.append(vector_angle(left, right))

            up = np.dot(np.linalg.inv(K), np.array([width / 2., 0, 1.]).reshape(3, 1))
            down = np.dot(np.linalg.inv(K), np.array([width / 2., height, 1.]).reshape(3, 1))
            img_updown_angles.append(vector_angle(up, down))

            img_angle_variations.append(max([img_leftright_angles[-1], img_updown_angles[-1]]))

            # all_rotations.append(R)
            # all_translations.append(t)

        # compute pair-wise triangulation angles
        cnt = len(cam_center_positions)
        plt.figure(figsize=(14, 8))
        plt.plot(range(0, cnt), img_angle_variations)
        plt.xticks(range(0, cnt, 1))
        plt.xlabel('image index')
        plt.grid(True)
        plt.ylabel('field of view (degrees)')
        plt.title('field of view')
        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, 'field_of_view.jpg'))
        plt.close()

        # # compute pair-wise forward motions
        # pairwise_motions = np.zeros((cnt, cnt))
        # for i in range(cnt):
        #     for j in range(cnt):
        #         # j with respect to i
        #         relative_translation = -np.dot(np.dot(all_rotations[i], all_rotations[j].T), all_translations[j]) + all_translations[i]
        #         pairwise_motions[i, j] = relative_translation[2, 0]
        # plt.figure(figsize=(14, 10))
        # plt.imshow(pairwise_motions, cmap='magma')
        # plt.colorbar()
        # plt.xticks(range(0, cnt, 1))
        # plt.xlabel('image index')
        # plt.yticks(range(0, cnt, 1))
        # plt.ylabel('image index')
        # plt.title('pairwise forward motions (meters)')
        # plt.tight_layout()
        # plt.savefig(os.path.join(self.out_dir, 'pairwise_forward_motions.jpg'))
        # plt.close()

        cam_center_angles = np.zeros((cnt, cnt))
        img_center_angles = np.zeros((cnt, cnt))
        for i in range(cnt):
            for j in range(i+1, cnt):
                cam_center_angles[i, j] = vector_angle(cam_center_positions[i], cam_center_positions[j])
                cam_center_angles[j, i] = cam_center_angles[i, j]

                img_center_angles[i, j] = vector_angle(img_center_rays[i], img_center_rays[j])
                img_center_angles[j, i] = img_center_angles[i, j]

        plt.figure(figsize=(14, 10))
        plt.imshow(cam_center_angles, cmap='magma')
        plt.colorbar()
        plt.xticks(range(0, cnt, 1))
        plt.xlabel('image index')
        plt.yticks(range(0, cnt, 1))
        plt.ylabel('image index')
        plt.title('camera_center_pairwise_angles (degrees)')
        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, 'camera_center_pairwise_angles.jpg'))
        plt.close()

        plt.figure(figsize=(14, 10))
        plt.imshow(img_center_angles, cmap='magma')
        plt.colorbar()
        plt.xticks(range(0, cnt, 1))
        plt.xlabel('image index')
        plt.yticks(range(0, cnt, 1))
        plt.ylabel('image index')
        plt.title('image_center_pairwise_angles (degrees)')
        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, 'image_center_pairwise_angles.jpg'))
        plt.close()


if __name__ == '__main__':
    pass
