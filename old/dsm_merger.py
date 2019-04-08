import numpy as np
import json
import os
from colmap.read_dense import read_array
from lib.save_image_only import save_image_only
from lib.ply_np_converter import np2ply
from lib.georegister_dense import georegister_dense
import cv2
from lib.image_util import read_image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt



class DsmMerger(object):
    def __init__(self, bbx, resolution):
        self.xoff, self.yoff, self.xsize, self.ysize = bbx
        self.resolution = resolution

        # average height
        self.mean_height = np.empty((self.ysize, self.xsize))
        self.mean_height.fill(np.nan)
        # max height
        self.max_height = np.empty((self.ysize, self.xsize))
        self.max_height.fill(np.nan)
        # min height
        self.min_height = np.empty((self.ysize, self.xsize))
        self.min_height.fill(np.nan)

        # average squared height
        self.mean_sq_height = np.empty((self.ysize, self.xsize))
        self.mean_sq_height.fill(np.nan)
        # number of pixels
        self.num_pixels = np.zeros((self.ysize, self.xsize))

        # number of dsm_merged
        self.merged_dsm_cnt = 0

    def get_merged_dsm_max(self):
        max_height = cv2.medianBlur(self.max_height.astype(dtype=np.float32), 3)

        return max_height

    def get_merged_dsm_mean(self):
        return self.mean_height

    def get_var_mask(self):
        return np.sqrt(self.mean_sq_height - self.mean_height ** 2)

    def get_empty_mask(self):
        return self.num_pixels < 1

    def get_merged_ply_max(self):
        x_pts = np.array([self.xoff + (i + 0.5) * self.resolution for i in range(self.xsize)])
        y_pts = np.array([self.yoff - (i + 0.5) * self.resolution for i in range(self.ysize)])
        yy, xx = np.meshgrid(y_pts, x_pts, indexing='ij')

        yy = np.reshape(yy, (-1, 1))
        xx = np.reshape(xx, (-1, 1))

        # perform median filter
        max_height = cv2.medianBlur(self.max_height.astype(dtype=np.float32), 3)

        zz = np.reshape(max_height, (-1, 1))
        mask = np.logical_not(np.isnan(zz))

        # select out valid values
        xx = xx[mask].reshape((-1, 1))
        yy = yy[mask].reshape((-1, 1))
        zz = zz[mask].reshape((-1, 1))

        return np.hstack((xx, yy, zz))

    def get_merged_ply_mean(self):
        x_pts = np.array([self.xoff + (i + 0.5) * self.resolution for i in range(self.xsize)])
        y_pts = np.array([self.yoff - (i + 0.5) * self.resolution for i in range(self.ysize)])
        yy, xx = np.meshgrid(y_pts, x_pts, indexing='ij')

        yy = np.reshape(yy, (-1, 1))
        xx = np.reshape(xx, (-1, 1))

        zz = np.reshape(self.mean_height, (-1, 1))
        mask = np.logical_not(np.isnan(zz))

        # select out valid values
        xx = xx[mask].reshape((-1, 1))
        yy = yy[mask].reshape((-1, 1))
        zz = zz[mask].reshape((-1, 1))

        return np.hstack((xx, yy, zz))

    def add(self, points):
        new_pixels = 0
        updated_pixels = 0

        current_dsm = np.empty((self.ysize, self.xsize))
        current_dsm.fill(np.nan)

        cnt = points.shape[0]
        for i in range(cnt):
            x = points[i, 0]
            y = points[i, 1]
            z = points[i, 2]

            # row index
            r = int(np.floor((self.yoff - y) / self.resolution))
            c = int(np.floor((x - self.xoff) / self.resolution))

            # whether lie inside the boundary
            if r < 0 or c < 0 or r >= self.ysize or c >= self.xsize:
                continue

            if np.isnan(self.mean_height[r, c]):
                new_pixels += 1

                self.mean_height[r, c] = z
                self.max_height[r, c] = z
                self.min_height[r, c] = z
                self.mean_sq_height[r, c] = z ** 2
            else:
                updated_pixels += 1
                # update
                num_pixel = self.num_pixels[r, c]
                self.mean_height[r, c] = (self.mean_height[r, c] * num_pixel + z) / (num_pixel + 1)
                self.mean_sq_height[r, c] = (self.mean_sq_height[r, c] * num_pixel + z ** 2) / (num_pixel + 1)

                if z > self.max_height[r, c]:
                    self.max_height[r, c] = z

                if z < self.min_height[r, c]:
                    self.min_height[r, c] = z

            # update number of pixels
            self.num_pixels[r, c] += 1

            # write to current dsm
            if np.isnan(current_dsm[r, c]):
                current_dsm[r, c] = z
            elif z > current_dsm[r, c]:
                current_dsm[r, c] = z

        self.merged_dsm_cnt += 1
        # print report
        total = self.xsize * self.ysize
        print('{}/{}% pixels added, {}/{}% pixels updated'.format(new_pixels, new_pixels / total * 100,
                                                                  updated_pixels, updated_pixels / total * 100))
        print('{} dsm merged till far'.format(self.merged_dsm_cnt))

        return current_dsm

# test function
def merge_dsm(work_dir):
    out_dir = os.path.join(work_dir, 'dsm_merger')
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    aoi_file = os.path.join(work_dir, 'aoi.json')
    with open(aoi_file) as fp:
        aoi_dict = json.load(fp)
    #
    # # change the origin to the lower-left bottom of the aoi
    # aoi_dict['x'] = 0
    # aoi_dict['y'] = aoi_dict['h']

    gt_file = os.path.join(work_dir, 'evaluation/eval_ground_truth.tif')
    gt_dsm, geo, proj, meta, width, height = read_image(gt_file)
    gt_dsm = gt_dsm[:, :, 0]

    bbx = (geo[0]-aoi_dict['x'], geo[3]-(aoi_dict['y'] - aoi_dict['h']), width, height)
    resolution = geo[1]
    dsm_merger = DsmMerger(bbx, resolution)

    mvs_dir = os.path.join(work_dir, 'colmap/mvs')
    # first load inv_proj_mats
    inv_proj_mats = {}
    with open(os.path.join(mvs_dir, 'inv_proj_mats.txt')) as fp:
        for line in fp.readlines():
            tmp = line.split(' ')
            img_name = tmp[0]
            mats = np.array([float(tmp[i]) for i in range(1, 17)]).reshape((4, 4))
            inv_proj_mats[img_name] = mats

    # then read the depth maps
    depth_dir = os.path.join(mvs_dir, 'stereo/depth_maps')
    for item in sorted(os.listdir(depth_dir)):
        depth_type = 'geometric'
        idx = item.rfind('.{}.bin'.format(depth_type))
        if idx == -1:
            continue

        img_name = item[:idx]
        depth_map = read_array(os.path.join(depth_dir, item))
        # create a meshgrid
        height, width = depth_map.shape
        col, row = np.meshgrid(range(width), range(height))

        col = col.reshape((1, -1))
        row = row.reshape((1, -1))

        depth = depth_map[row, col]

        depth = depth.reshape((1, -1))

        mask = depth > 0

        tmp = np.vstack((col+0.5, row+0.5, np.ones((1, width * height)), 1.0 / depth))
        tmp = np.dot(inv_proj_mats[img_name], tmp)

        xx = tmp[0:1, :] / tmp[3:4, :]
        yy = tmp[1:2, :] / tmp[3:4, :]
        zz = tmp[2:3, :] / tmp[3:4, :]

        xx = xx[mask].reshape((-1, 1))
        yy = yy[mask].reshape((-1, 1))
        zz = zz[mask].reshape((-1, 1))

        points = np.hstack((xx, yy, zz))
        current_dsm = dsm_merger.add(points)

        # write to
        disp_depth_min = 10
        disp_depth_max = 50
        current_dsm[np.isnan(current_dsm)] = disp_depth_min
        current_dsm[current_dsm < disp_depth_min] = disp_depth_min
        current_dsm[current_dsm > disp_depth_max] = disp_depth_max
        # trick here
        current_dsm[0, 0] = disp_depth_min
        current_dsm[0, 1] = disp_depth_max
        save_image_only(current_dsm, os.path.join(out_dir, '{}.dsm.jpg'.format(img_name)))

    dsm = dsm_merger.get_merged_dsm_max()

    diff_dsm = np.abs(dsm - gt_dsm)
    diff_dsm[gt_dsm < -9000] = 0

    diff_dsm[diff_dsm > 5] = 5
    mask = np.logical_and(gt_dsm > -9000, np.isnan(dsm))
    diff_dsm[mask] = 6

    save_image_only(diff_dsm, os.path.join(out_dir, 'dsm_diff.jpg'))

    plt.figure()
    plt.imshow(diff_dsm, cmap='magma')
    plt.colorbar()
    plt.savefig(os.path.join(out_dir, 'dsm_diff_with_colorbar.jpg'))
    plt.close()

    disp_depth_min = 10
    disp_depth_max = 50

    dsm[np.isnan(dsm)] = disp_depth_min
    dsm[dsm < disp_depth_min] = disp_depth_min
    dsm[dsm > disp_depth_max] = disp_depth_max
    save_image_only(dsm, os.path.join(out_dir, 'dsm_merged.jpg'))

    var_mask = dsm_merger.get_var_mask()
    var_mask[np.isnan(var_mask)] = 0
    var_mask[var_mask > 2] = 2
    save_image_only(var_mask, os.path.join(out_dir, 'dsm_stddev.jpg'))

    plt.figure()
    plt.imshow(var_mask, cmap='magma')
    plt.colorbar()
    plt.savefig(os.path.join(out_dir, 'dsm_stddev_with_colorbar.jpg'))
    plt.close()

    # point_cloud = dsm_merger.get_merged_ply_max()
    # np2ply(point_cloud, os.path.join(work_dir, 'point_cloud.ply'))
    #
    # M = np.identity(3)
    # t = np.zeros((1, 3))
    # with open(os.path.join(work_dir, 'aoi.json')) as fp:
    #     aoi_dict = json.load(fp)
    # aoi_ll_east = aoi_dict['x']
    # aoi_ll_north = aoi_dict['y'] - aoi_dict['h']
    # t[0, 0] += aoi_ll_east
    # t[0, 1] += aoi_ll_north
    #
    # georegister_dense(os.path.join(work_dir, 'point_cloud.ply'),
    #                   os.path.join(work_dir, 'evaluation/eval_point_cloud.ply'),
    #                   os.path.join(work_dir, 'aoi.json'), M, t, filter=True)


if __name__ == '__main__':
    work_dir = '/data2/kz298/mvs3dm_result/MasterSequesteredPark'

    merge_dsm(work_dir)
