import numpy as np


# x axis points right
# y axis points down
def proj_to_geo_grid(points, xoff, yoff, resolution, xsize, ysize):
    current_dsm = np.empty((ysize, xsize))
    current_dsm.fill(np.nan)

    cnt = points.shape[0]
    for i in range(cnt):
        x = points[i, 0]
        y = points[i, 1]
        z = points[i, 2]

        # row index
        # half pixel
        r = int(np.floor((yoff - y) / resolution) + 0.5)
        c = int(np.floor((x - xoff) / resolution) + 0.5)

        # whether lie inside the boundary
        if r < 0 or c < 0 or r >= ysize or c >= xsize:
            continue

        # write to current dsm
        if np.isnan(current_dsm[r, c]):
            current_dsm[r, c] = z
        elif z > current_dsm[r, c]:
            current_dsm[r, c] = z

    print('current_dsm empty ratio: {}%'.format(
        np.sum(np.isnan(current_dsm)) / current_dsm.size * 100.))

    return current_dsm
