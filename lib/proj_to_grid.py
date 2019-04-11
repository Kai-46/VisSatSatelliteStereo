import numpy as np


# points: (xx, yy, zz)
# xoff: ul_easting
# yoff: ul_northing
# xsize: width
# ysize: height
def proj_to_grid(points, xoff, yoff, xresolution, yresolution, xsize, ysize, propagate=False):
    dsm = np.empty((ysize, xsize))
    dsm.fill(np.nan)

    cnt = points.shape[0]
    for i in range(cnt):
        x = points[i, 0]
        y = points[i, 1]
        z = points[i, 2]

        # row index
        # half pixel
        r = int(np.floor((yoff - y) / xresolution))
        c = int(np.floor((x - xoff) / yresolution))

        # whether lie inside the boundary
        if r < 0 or c < 0 or r >= ysize or c >= xsize:
            continue

        # write to current dsm
        if np.isnan(dsm[r, c]):
            dsm[r, c] = z
        elif z > dsm[r, c]:     # take the maximum value
            dsm[r, c] = z

        # modify neighbors
        if propagate:
            if r - 1 >= 0 and np.isnan(dsm[r - 1, c]):
                dsm[r - 1, c] = z

            if r + 1 < ysize and np.isnan(dsm[r + 1, c]):
                dsm[r + 1, c] = z

            if c - 1 >= 0 and np.isnan(dsm[r, c - 1]):
                dsm[r, c - 1] = z

            if c + 1 < xsize and np.isnan(dsm[r, c + 1]):
                dsm[r, c + 1] = z

    # print('dsm empty ratio: {}%'.format(
    #     np.sum(np.isnan(dsm)) / dsm.size * 100.))

    return dsm
