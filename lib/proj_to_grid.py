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

# import numpy as np
#
#
# # points: (xx, yy, zz)
# # xoff: ul_easting
# # yoff: ul_northing
# # xsize: width
# # ysize: height
# def proj_to_grid(points, xoff, yoff, xresolution, yresolution, xsize, ysize, propagate=False):
#     dsm = np.empty((ysize, xsize))
#     dsm.fill(np.nan)
#
#     cnt = points.shape[0]
#     for i in range(cnt):
#         x = points[i, 0]
#         y = points[i, 1]
#         z = points[i, 2]
#
#         # row index
#         # half pixel
#         r = int(np.floor((yoff - y) / xresolution))
#         c = int(np.floor((x - xoff) / yresolution))
#
#         # whether lie inside the boundary
#         if r < 0 or c < 0 or r >= ysize or c >= xsize:
#             continue
#
#         # write to current dsm
#         if np.isnan(dsm[r, c]):
#             dsm[r, c] = z
#         elif z > dsm[r, c]:     # take the maximum value
#             dsm[r, c] = z
#
#         # modify neighbors
#         # if propagate:
#         #     if r - 1 >= 0 and np.isnan(dsm[r - 1, c]):
#         #         dsm[r - 1, c] = z
#         #
#         #     if r + 1 < ysize and np.isnan(dsm[r + 1, c]):
#         #         dsm[r + 1, c] = z
#         #
#         #     if c - 1 >= 0 and np.isnan(dsm[r, c - 1]):
#         #         dsm[r, c - 1] = z
#         #
#         #     if c + 1 < xsize and np.isnan(dsm[r, c + 1]):
#         #         dsm[r, c + 1] = z
#
#     # print('dsm empty ratio: {}%'.format(
#     #     np.sum(np.isnan(dsm)) / dsm.size * 100.))
#
#     # try to fill very small holes
#     dsm_new = dsm.copy()
#     nan_places = np.argwhere(np.isnan(dsm_new))
#     for i in range(nan_places.shape[0]):
#         row = nan_places[i, 0]
#         col = nan_places[i, 1]
#         neighbors = []
#         for j in range(row-1, row+2):
#             for k in range(col-1, col+2):
#                 if j >= 0 and j < dsm_new.shape[0] and k >=0 and k < dsm_new.shape[1]:
#                     val = dsm_new[j, k]
#                     if not np.isnan(val):
#                         neighbors.append(val)
#
#         if neighbors:
#             dsm[row, col] = np.median(neighbors)
#
#     return dsm


import numpy as np
import numpy_groupies as npg


# points: each row is (xx, yy, zz)
# xoff: ul_e
# yoff: ul_n
# xsize: width
# ysize: height
def proj_to_grid(points, xoff, yoff, xresolution, yresolution, xsize, ysize, propagate=False):
    row = np.floor((yoff - points[:, 1]) / xresolution).astype(dtype=np.int)
    col = np.floor((points[:, 0] - xoff) / yresolution).astype(dtype=np.int)
    points_group_idx = row * xsize + col
    points_val = points[:, 2]

    # remove points that lie out of the dsm boundary
    mask = ((row >= 0) * (col >= 0) * (row < ysize) * (col < xsize)) > 0
    points_group_idx = points_group_idx[mask]
    points_val = points_val[mask]

    # create a place holder for all pixels in the dsm
    group_idx = np.arange(xsize * ysize).astype(dtype=np.int)
    group_val = np.empty(xsize * ysize)
    group_val.fill(np.nan)

    # concatenate place holders with the real valuies, then aggregate
    group_idx = np.concatenate((group_idx, points_group_idx))
    group_val = np.concatenate((group_val, points_val))

    dsm = npg.aggregate(group_idx, group_val, func='nanmax', fill_value=np.nan)
    dsm = dsm.reshape((ysize, xsize))

    # try to fill very small holes
    dsm_new = dsm.copy()
    nan_places = np.argwhere(np.isnan(dsm_new))
    for i in range(nan_places.shape[0]):
        row = nan_places[i, 0]
        col = nan_places[i, 1]
        neighbors = []
        for j in range(row-1, row+2):
            for k in range(col-1, col+2):
                if j >= 0 and j < dsm_new.shape[0] and k >=0 and k < dsm_new.shape[1]:
                    val = dsm_new[j, k]
                    if not np.isnan(val):
                        neighbors.append(val)

        if neighbors:
            dsm[row, col] = np.median(neighbors)

    return dsm
