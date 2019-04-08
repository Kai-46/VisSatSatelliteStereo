import numpy as np
from lib.ply_np_converter import np2ply, ply2np
import multiprocessing
import os


class MaxGrid(object):
    # note that bbx is specified as [x_min, x_max, y_min, y_max]
    # assume (x_min, y_min) is the lower-left corner, and x axis points to the right, y points to up
    def __init__(self, bbx, spacing):
        self.spacing = spacing
        self.x_min = bbx[0]
        self.y_min = bbx[2]
        self.w = int((bbx[1] - bbx[0]) / self.spacing) + 1
        self.h = int((bbx[3] - bbx[2]) / self.spacing) + 1

        self.grid = np.empty((self.h, self.w))
        self.grid.fill(-np.inf)


    def build_grid(self, data):
        for i in range(data.shape[0]):
            x = data[i, 0]
            y = data[i, 1]
            z = data[i, 2]

            # compute idx
            col_idx = int((x - self.x_min) / self.spacing)
            row_idx = self.h - 1 - int((y - self.y_min) / self.spacing)

            if z > self.grid[row_idx, col_idx]:
                self.grid[row_idx, col_idx] = z

    def build_grid_fill(self, data):
        # let's first try to build a data structure that register the points
        register = [[[] for _ in range(self.w)] for _ in range(self.h)]
        coarse_register = [[[] for _ in range(self.w)] for _ in range(self.h)]

        for i in range(data.shape[0]):
            x = data[i, 0]
            y = data[i, 1]
            z = data[i, 2]

            # compute idx
            col_idx = int((x - self.x_min) / self.spacing)
            row_idx = self.h - 1 - int((y - self.y_min) / self.spacing)

            if col_idx < 0 or col_idx >= self.w or row_idx < 0 or row_idx >= self.h:
                continue

            register[row_idx][col_idx].append(z)

            # check whether we should register to the adjacent grid points
            if row_idx > 1:
                coarse_register[row_idx-1][col_idx].append(z)
            if row_idx < self.h - 1:
                coarse_register[row_idx+1][col_idx].append(z)
            if col_idx > 1:
                coarse_register[row_idx][col_idx-1].append(z)
            if col_idx < self.w - 1:
                coarse_register[row_idx][col_idx+1].append(z)

        for i in range(self.h):
            for j in range(self.w):
                tmp = register[i][j]
                if len(tmp) > 0:
                    self.grid[i, j] = max(tmp)
                else:
                    tmp = coarse_register[i][j]
                    if len(tmp) > 0:
                        self.grid[i, j] = max(tmp)

    # x --> right, y --> up
    # def get_grid_point(self, x, y):
    #     # compute idx
    #     col_idx = int((x - self.x_min) / self.spacing)
    #     row_idx = self.h - 1 - int((y - self.y_min) / self.spacing)
    #
    #     return self.grid[row_idx, col_idx]

    def grid_points(self):
        points = []
        for i in range(self.h):
            for j in range(self.w):
                if self.grid[i, j] > -1e10:
                    center_x = self.x_min + 0.5 * self.spacing + j * self.spacing
                    center_y = self.y_min + 0.5 * self.spacing + (self.h - 1 - i) * self.spacing
                    points.append([center_x, center_y, self.grid[i, j]])
        return np.array(points)


def run_flatten(data, bbx, spacing, result_file):
    grid = MaxGrid(bbx, spacing)
    grid.build_grid_fill(data)
    out_data = grid.grid_points()

    np.save(result_file, out_data)


def flatten_point_cloud(in_ply_file, spacing, out_ply_file, bbx=None, tif_file=None):
    in_ply_data, comments = ply2np(in_ply_file, return_comments=True)

    if bbx is None:
        bbx = (np.min(in_ply_data[:, 0]), np.max(in_ply_data[:, 0]), np.min(in_ply_data[:, 1]), np.max(in_ply_data[:, 1]))

    # grid = MaxGrid(bbx, spacing)
    # out_ply_data = grid.build_grid_fill(in_ply_data[:, 0:3]).grid_points()
    # out_ply_data = np.hstack((out_ply_data, in_ply_data[:, 3:]))

    process_cnt = 9
    process_list = []
    # 3 \ 3 bbx
    x_cut = np.linspace(bbx[0], bbx[1], 3+1)
    y_cut = np.linspace(bbx[2], bbx[3], 3+1)
    sub_bbx = []
    for i in range(3):
        for j in range(3):
            sub_bbx.append([x_cut[i], x_cut[i+1], y_cut[j], y_cut[j+1]])

    result_file_list = []
    for i in range(process_cnt):
        result_file = '{}_{}.npy'.format(out_ply_file, i)
        p = multiprocessing.Process(target=run_flatten, args=(in_ply_data[:, 0:3], sub_bbx[i], spacing, result_file))
        result_file_list.append(result_file)
        process_list.append(p)
        p.start()

    for p in process_list:
        p.join()

    data = []
    for result_file in result_file_list:
        data.append(np.load(result_file))
        os.remove(result_file)

    out_ply_data = np.vstack(tuple(data))
    np2ply(out_ply_data, out_ply_file, comments)


if __name__ == '__main__':
    spacing = 0.3
    flatten_point_cloud('/data2/kz298/mvs3dm_result/MasterSequestered1/evaluation/eval_point_cloud.ply', spacing,
                        '/data2/kz298/mvs3dm_result/MasterSequestered1/evaluation/eval_point_cloud_new.ply')