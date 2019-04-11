import numpy as np
from lib.latlon_utm_converter import latlon_to_eastnorh
from lib.proj_to_grid import proj_to_grid


# points: (lat, lon, alt)
# xoff: ul_easting
# yoff: ul_northing
# xsize: width
# ysize: height
def proj_to_utm_grid(points, xoff, yoff, xresolution, yresolution, xsize, ysize, propagate=False):
    east, north = latlon_to_eastnorh(points[:, 0:1], points[:, 1:2])
    points = np.hstack((east, north, points[:, 2:3]))

    dsm = proj_to_grid(points, xoff, yoff, xresolution, yresolution, xsize, ysize, propagate)

    return dsm
