import numpy as np
import json
import os
from lib.georegister_dense import georegister_dense
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


from matplotlib.colors import BoundaryNorm
# a=np.random.randn(2500).reshape((50,50))

# define the colormap
cmap = plt.get_cmap('PuOr')

# extract all colors from the .jet map
cmaplist = [cmap(i) for i in range(cmap.N)]
# create the new map
cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)

# define the bins and normalize and forcing 0 to be part of the colorbar!
bounds = np.arange(-5, 5, .1)
idx=np.searchsorted(bounds,0)
bounds=np.insert(bounds,idx,0)
norm = BoundaryNorm(bounds, cmap.N)


class DsmMerger(object):
    def __init__(self, bbx, resolution):
        self.xoff, self.yoff, self.xsize, self.ysize = bbx
        self.resolution = resolution

        self.all_dsm = []

        # mask
        self.empty_mask = np.empty((self.ysize, self.xsize))
        self.empty_mask.fill(True)

    # median absolute deviation
    def get_mad_mask(self):
        all_dsm = np.concatenate(tuple([dsm[:, :, np.newaxis] for dsm in self.all_dsm]), axis=2)

        # number of dsm
        dsm_cnt = all_dsm.shape[2]
        # compute median
        tmp = np.nanmedian(all_dsm, axis=2, keepdims=True)
        tmp = np.abs(all_dsm - np.tile(tmp, (1, 1, dsm_cnt)))

        # compute mad
        return np.nanmedian(tmp, axis=2)

    def get_merged_dsm(self):
        all_dsm = np.concatenate(tuple([dsm[:, :, np.newaxis] for dsm in self.all_dsm]), axis=2)
        # number of dsm
        dsm_cnt = all_dsm.shape[2]
        # compute median
        tmp = np.nanmedian(all_dsm, axis=2, keepdims=True)
        tmp = np.abs(all_dsm - np.tile(tmp, (1, 1, dsm_cnt)))
        mad = np.nanmedian(tmp, axis=2, keepdims=True)

        mask = tmp > np.tile(mad, (1, 1, dsm_cnt))
        all_dsm[mask] = np.nan

        return np.nanmean(all_dsm, axis=2)

    def get_merged_dsm_max(self):
        all_dsm = np.concatenate(tuple([dsm[:, :, np.newaxis] for dsm in self.all_dsm]), axis=2)

        max_height = np.nanmax(all_dsm, axis=2)

        #max_height = cv2.medianBlur(max_height.astype(dtype=np.float32), 3)

        return max_height

    def get_merged_dsm_mean(self):
        all_dsm = np.concatenate(tuple([dsm[:, :, np.newaxis] for dsm in self.all_dsm]), axis=2)

        mean_height = np.nanmean(all_dsm, axis=2)
        return mean_height

    def get_merged_dsm_median(self):
        all_dsm = np.concatenate(tuple([dsm[:, :, np.newaxis] for dsm in self.all_dsm]), axis=2)

        median_height = np.nanmedian(all_dsm, axis=2)

        return median_height

    def get_stddev_mask(self):
        all_dsm = np.concatenate(tuple([dsm[:, :, np.newaxis] for dsm in self.all_dsm]), axis=2)

        return np.nanstd(all_dsm, axis=2)

    def get_num_measure_mask(self):
        all_dsm = np.concatenate(tuple([dsm[:, :, np.newaxis] for dsm in self.all_dsm]), axis=2)
        return np.sum(np.logical_not(np.isnan(all_dsm)), axis=2)

    def get_empty_mask(self):
        all_dsm = np.concatenate(tuple([dsm[:, :, np.newaxis] for dsm in self.all_dsm]), axis=2)

        # double check
        # mask = np.zeros(self.all_dsm[0].shape) < 1
        # for dsm in self.all_dsm:
        #     mask = np.logical_and(mask, np.isnan(dsm))
        # print('here empty ratio: {}%'.format(np.sum(mask)/mask.size*100.))

        return np.all(np.isnan(all_dsm), axis=2)

    def add(self, points):
        current_dsm = np.empty((self.ysize, self.xsize))
        current_dsm.fill(np.nan)

        cnt = points.shape[0]
        for i in range(cnt):
            x = points[i, 0]
            y = points[i, 1]
            z = points[i, 2]

            # row index
            # half pixel
            r = int(np.floor((self.yoff - y) / self.resolution) + 0.5)
            c = int(np.floor((x - self.xoff) / self.resolution) + 0.5)

            # whether lie inside the boundary
            if r < 0 or c < 0 or r >= self.ysize or c >= self.xsize:
                continue

            # write to current dsm
            if np.isnan(current_dsm[r, c]):
                current_dsm[r, c] = z
            elif z > current_dsm[r, c]:
                current_dsm[r, c] = z

        # must explicitly copy here
        self.all_dsm.append(np.copy(current_dsm))
        print('current_dsm empty ratio: {}%, {} dsm merged till far'.format(
            np.sum(np.isnan(current_dsm))/current_dsm.size*100., len(self.all_dsm)))

        # update mask
        mask = np.isnan(current_dsm)
        empty_ratio = np.sum(self.empty_mask) / self.empty_mask.size
        self.empty_mask = np.logical_and(self.empty_mask, mask)
        new_empty_ratio = np.sum(self.empty_mask) / self.empty_mask.size
        print('global empty ratio: {}% --> {}%'.format(empty_ratio * 100, new_empty_ratio * 100))

        return current_dsm

    def convert_to_ply(self, dsm):
        x_pts = np.array([self.xoff + i * self.resolution for i in range(self.xsize)])
        y_pts = np.array([self.yoff - i * self.resolution for i in range(self.ysize)])
        yy, xx = np.meshgrid(y_pts, x_pts, indexing='ij')

        yy = np.reshape(yy, (-1, 1))
        xx = np.reshape(xx, (-1, 1))

        # perform median filter
        zz = np.reshape(dsm, (-1, 1))
        mask = np.logical_not(np.isnan(zz))

        # select out valid values
        xx = xx[mask].reshape((-1, 1))
        yy = yy[mask].reshape((-1, 1))
        zz = zz[mask].reshape((-1, 1))

        return np.hstack((xx, yy, zz))

    # def get_merged_ply_max(self):
    #     x_pts = np.array([self.xoff + (i + 0.5) * self.resolution for i in range(self.xsize)])
    #     y_pts = np.array([self.yoff - (i + 0.5) * self.resolution for i in range(self.ysize)])
    #     yy, xx = np.meshgrid(y_pts, x_pts, indexing='ij')
    #
    #     yy = np.reshape(yy, (-1, 1))
    #     xx = np.reshape(xx, (-1, 1))
    #
    #     # perform median filter
    #     max_height = cv2.medianBlur(self.max_height.astype(dtype=np.float32), 3)
    #
    #     zz = np.reshape(max_height, (-1, 1))
    #     mask = np.logical_not(np.isnan(zz))
    #
    #     # select out valid values
    #     xx = xx[mask].reshape((-1, 1))
    #     yy = yy[mask].reshape((-1, 1))
    #     zz = zz[mask].reshape((-1, 1))
    #
    #     return np.hstack((xx, yy, zz))
    #
    # def get_merged_ply_mean(self):
    #     x_pts = np.array([self.xoff + (i + 0.5) * self.resolution for i in range(self.xsize)])
    #     y_pts = np.array([self.yoff - (i + 0.5) * self.resolution for i in range(self.ysize)])
    #     yy, xx = np.meshgrid(y_pts, x_pts, indexing='ij')
    #
    #     yy = np.reshape(yy, (-1, 1))
    #     xx = np.reshape(xx, (-1, 1))
    #
    #     zz = np.reshape(self.mean_height, (-1, 1))
    #     mask = np.logical_not(np.isnan(zz))
    #
    #     # select out valid values
    #     xx = xx[mask].reshape((-1, 1))
    #     yy = yy[mask].reshape((-1, 1))
    #     zz = zz[mask].reshape((-1, 1))
    #
    #     return np.hstack((xx, yy, zz))


# test function
def merge_dsm(work_dir):
    out_dir = os.path.join(work_dir, 'dsm_merger')
    # if not os.path.exists(out_dir):
    #     os.mkdir(out_dir)
    #
    # aoi_file = os.path.join(work_dir, 'aoi.json')
    # with open(aoi_file) as fp:
    #     aoi_dict = json.load(fp)
    # #
    # # # change the origin to the lower-left bottom of the aoi
    # # aoi_dict['x'] = 0
    # # aoi_dict['y'] = aoi_dict['h']
    #
    # gt_file = os.path.join(work_dir, 'evaluation/eval_ground_truth.tif')
    # gt_dsm, geo, proj, meta, width, height = read_image(gt_file)
    # gt_dsm = gt_dsm[:, :, 0]
    #
    # bbx = (geo[0]-aoi_dict['ul_easting'], geo[3]-(aoi_dict['ul_northing'] - aoi_dict['height']), width, height)
    # resolution = geo[1]
    # dsm_merger = DsmMerger(bbx, resolution)
    #
    # mvs_dir = os.path.join(work_dir, 'colmap/mvs')
    # # first load inv_proj_mats
    # inv_proj_mats = {}
    # with open(os.path.join(mvs_dir, 'inv_proj_mats.txt')) as fp:
    #     for line in fp.readlines():
    #         tmp = line.split(' ')
    #         img_name = tmp[0]
    #         mats = np.array([float(tmp[i]) for i in range(1, 17)]).reshape((4, 4))
    #         inv_proj_mats[img_name] = mats
    #
    # # then read the depth maps
    # depth_dir = os.path.join(mvs_dir, 'stereo/depth_maps')
    # cnt = 0
    # for item in sorted(os.listdir(depth_dir)):
    #     depth_type = 'geometric'
    #     idx = item.rfind('.{}.bin'.format(depth_type))
    #     if idx == -1:
    #         continue
    #
    #     img_name = item[:idx]
    #     depth_map = read_array(os.path.join(depth_dir, item))
    #     # create a meshgrid
    #     height, width = depth_map.shape
    #     col, row = np.meshgrid(range(width), range(height))
    #
    #     col = col.reshape((1, -1))
    #     row = row.reshape((1, -1))
    #
    #     depth = depth_map[row, col]
    #
    #     depth = depth.reshape((1, -1))
    #
    #     mask = depth > 0
    #
    #     tmp = np.vstack((col, row, np.ones((1, width * height)), 1.0 / depth))
    #     tmp = np.dot(inv_proj_mats[img_name], tmp)
    #
    #     xx = tmp[0:1, :] / tmp[3:4, :]
    #     yy = tmp[1:2, :] / tmp[3:4, :]
    #     zz = tmp[2:3, :] / tmp[3:4, :]
    #
    #     xx = xx[mask].reshape((-1, 1))
    #     yy = yy[mask].reshape((-1, 1))
    #     zz = zz[mask].reshape((-1, 1))
    #
    #     points = np.hstack((xx, yy, zz))
    #     current_dsm = dsm_merger.add(points)
    #
    #     cnt += 1
    #     # # debug
    #     # if cnt >= 3:
    #     #     break
    #
    #     # plot
    #     # disp_depth_min = 10
    #     # disp_depth_max = 50
    #     # current_dsm[np.isnan(current_dsm)] = disp_depth_min
    #     # current_dsm[current_dsm < disp_depth_min] = disp_depth_min
    #     # current_dsm[current_dsm > disp_depth_max] = disp_depth_max
    #     # # trick here, make sure the color is the same across images
    #     # current_dsm[0, 0] = disp_depth_min
    #     # current_dsm[0, 1] = disp_depth_max
    #
    #     current_dsm_empty_mask = np.isnan(current_dsm)
    #     save_image_only(current_dsm_empty_mask.astype(dtype=np.float),
    #                     os.path.join(out_dir, '{}.dsm.empty_mask.jpg'.format(img_name)))
    #     current_dsm[current_dsm_empty_mask] = np.nanmin(current_dsm) - 2.
    #     save_image_only(current_dsm, os.path.join(out_dir, '{}.dsm.jpg'.format(img_name)))
    #
    # dsm = dsm_merger.get_merged_dsm_median()
    # # convert to point cloud
    # point_cloud = dsm_merger.convert_to_ply(dsm)
    # np2ply(point_cloud, os.path.join(out_dir, 'point_cloud.ply'))
    #
    # # diff_dsm = np.abs(dsm - gt_dsm)
    # diff_dsm = dsm - gt_dsm
    # diff_dsm[gt_dsm < -9000] = 0
    #
    # diff_dsm[diff_dsm < -5] = -5
    # diff_dsm[diff_dsm > 5] = 5
    # mask = np.logical_and(gt_dsm > -9000, np.isnan(dsm))
    # diff_dsm[mask] = 5
    #
    # save_image_only(diff_dsm, os.path.join(out_dir, 'dsm_diff.jpg'), cmap=cmap, norm=norm)
    #
    # plt.figure()
    # plt.imshow(diff_dsm, cmap=cmap, norm=norm)
    # plt.colorbar()
    # plt.savefig(os.path.join(out_dir, 'dsm_diff_with_colorbar.jpg'))
    # plt.close()
    #
    # disp_depth_min = 10
    # disp_depth_max = 50
    #
    # dsm[np.isnan(dsm)] = disp_depth_min
    # dsm[dsm < disp_depth_min] = disp_depth_min
    # dsm[dsm > disp_depth_max] = disp_depth_max
    # save_image_only(dsm, os.path.join(out_dir, 'dsm_merged.jpg'))
    #
    # # plot a lot of masks
    # num_measure_mask = dsm_merger.get_num_measure_mask()
    #
    # print('single measurement: {}%'.format(np.sum(num_measure_mask == 1) / num_measure_mask.size * 100))
    # print('two measurements: {}%'.format(np.sum(num_measure_mask == 2) / num_measure_mask.size * 100))
    # single_measure_mask = (num_measure_mask == 1)
    # save_image_only(single_measure_mask.astype(np.float), os.path.join(out_dir, 'dsm_single_measure.jpg'))
    #
    # two_measure_mask = (num_measure_mask == 2)
    # save_image_only(two_measure_mask.astype(np.float), os.path.join(out_dir, 'dsm_two_measure.jpg'))
    #
    # save_image_only(num_measure_mask, os.path.join(out_dir, 'dsm_num_measure.jpg'))
    # plt.figure()
    # plt.imshow(num_measure_mask, cmap='magma')
    # plt.colorbar()
    # plt.savefig(os.path.join(out_dir, 'dsm_num_measure_with_colorbar.jpg'))
    # plt.close()
    #
    # stddev_mask = dsm_merger.get_stddev_mask()
    # stddev_mask[np.isnan(stddev_mask)] = 0
    # stddev_mask[stddev_mask > 2] = 2
    #
    # save_image_only(stddev_mask, os.path.join(out_dir, 'dsm_stddev.jpg'))
    #
    # plt.figure()
    # plt.imshow(stddev_mask, cmap='magma')
    # plt.colorbar()
    # plt.savefig(os.path.join(out_dir, 'dsm_stddev_with_colorbar.jpg'))
    # plt.close()
    #
    # empty_mask = dsm_merger.get_empty_mask()
    # print("\nfinal empty ratio: {}%".format(np.sum(empty_mask)/empty_mask.size*100))
    # save_image_only(empty_mask.astype(dtype=np.float), os.path.join(out_dir, 'dsm_empty_mask.jpg'))
    #
    # mad_mask = dsm_merger.get_mad_mask()
    #
    # # remove single-measurement mad
    # mad_mask[num_measure_mask < 2] = np.nan
    #
    # # plot distribution
    # mad_mask_vals = mad_mask[np.logical_not(np.isnan(mad_mask))]
    # # mad_mask_vals[mad_mask_vals > 1] = 1
    # mad_mask_vals = mad_mask_vals[mad_mask_vals < 1]
    # plt.figure(figsize=(14, 5), dpi=80)
    # plt.hist(mad_mask_vals, bins=50, density=True, cumulative=False)
    # max_mad_val = np.max(mad_mask_vals)
    # plt.xticks(np.arange(0, max_mad_val + 0.01, 0.1))
    # plt.xlabel('MAD statistic in each geo-grid cell for multi-view measurement')
    # plt.ylabel('pdf')
    # plt.title(
    #     '# of views: {}, geo-grid size: {}*{}, total # of valid cells: {}/{:.3f}%\nMAD (meters): median {:.3f}'
    #     .format(cnt, mad_mask.shape[1], mad_mask.shape[0], mad_mask_vals.size,
    #             mad_mask_vals.size/mad_mask.size*100.,
    #             np.median(mad_mask_vals)))
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig(os.path.join(out_dir, 'dsm_mad_dist.jpg'))
    # plt.close('all')
    #
    # # for visualization
    # mad_mask[np.isnan(mad_mask)] = 0
    # mad_mask[mad_mask > 2] = 2
    #
    # save_image_only(mad_mask, os.path.join(out_dir, 'dsm_mad.jpg'))
    #
    # plt.figure()
    # plt.imshow(mad_mask, cmap='magma')
    # plt.colorbar()
    # plt.savefig(os.path.join(out_dir, 'dsm_mad_with_colorbar.jpg'))
    # plt.close()

    M = np.identity(3)
    t = np.zeros((1, 3))
    with open(os.path.join(work_dir, 'aoi.json')) as fp:
        aoi_dict = json.load(fp)
    aoi_ll_east = aoi_dict['ul_easting']
    aoi_ll_north = aoi_dict['ul_northing'] - aoi_dict['height']
    t[0, 0] += aoi_ll_east
    t[0, 1] += aoi_ll_north

    georegister_dense(os.path.join(out_dir, 'point_cloud.ply'),
                      os.path.join(work_dir, 'evaluation/eval_point_cloud.ply'),
                      os.path.join(work_dir, 'aoi.json'), M, t, filter=True)


if __name__ == '__main__':
    work_dir = '/data2/kz298/mvs3dm_result/MasterSequesteredPark'

    merge_dsm(work_dir)
