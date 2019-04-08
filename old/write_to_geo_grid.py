import numpy as np
import cv2


def fill_in_small_holes(dsm):
    # set the empty pixels to be the background
    empty_val = np.nanmin(dsm) - 50
    dsm[np.isnan(dsm)] = empty_val
    # now dilate the pixels
    kernel = np.ones((3, 3), dtype=np.uint8)
    #kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    # dsm = cv2.dilate(dsm, kernel)

    #dsm = cv2.morphologyEx(dsm, cv2.MORPH_CLOSE, kernel)

    # dsm = cv2.medianBlur(dsm.astype(dtype=np.float32), 3)
    return dsm


def write_to_geo_grid(points, resolution, bbx=None):
    if bbx is not None:
        xoff, yoff, xsize, ysize = bbx
    else:
        xmin = np.min(points[:, 0])
        xmax = np.max(points[:, 0])
        ymin = np.min(points[:, 1])
        ymax = np.max(points[:, 1])

        xoff = xmin
        yoff = ymax
        xsize = int(1 + np.floor((xmax - xmin) / resolution))
        ysize = int(1 + np.floor((ymax - ymin) / resolution))

    geo_grid = np.empty((ysize, xsize))
    geo_grid.fill(np.nan)

    cnt = points.shape[0]
    for i in range(cnt):
        x = points[i, 0]
        y = points[i, 1]
        z = points[i, 2]

        # row index
        r = int(np.floor((yoff - y) / resolution))
        c = int(np.floor((x - xoff) / resolution))

        if np.isnan(geo_grid[r, c]):
            geo_grid[r, c] = z
        elif z > geo_grid[r, c]:
            geo_grid[r, c] = z

    geo_info = (xoff, yoff, xsize, ysize, resolution)
    return geo_grid, geo_info
