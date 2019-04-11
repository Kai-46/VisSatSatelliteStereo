from old.image_util import read_image, parse_proj_str
import numpy as np
from lib.latlon_utm_converter import eastnorth_to_latlon

# each pixel of a dsm is (east, north, height)
# return (east, north, height) and (latitude, longitude, height)
def dsm2np(dsm_file, nodata=-9999):
    dsm, geo, proj, meta, width, height = read_image(dsm_file)

    meta_dict = {
        'geo': geo,
        'proj': proj,
        'meta': meta,
        'width': width,
        'height': height
    }

    zone_number, hemisphere = parse_proj_str(proj)
    dsm = dsm[:, :, 0]

    ul_east = geo[0]
    ul_north = geo[3]
    east_resolution = geo[1]
    north_resolution = abs(geo[5])

    east_pts = np.array([ul_east + (i + 0.5) * east_resolution for i in range(width)])
    north_pts = np.array([ul_north - (i - 0.5) * north_resolution for i in range(height)])
    yy, xx = np.meshgrid(north_pts, east_pts, indexing='ij')

    yy = np.reshape(yy, (-1, 1))
    xx = np.reshape(xx, (-1, 1))
    zz = np.reshape(dsm, (-1, 1))

    # select out valid values
    mask = (zz == nodata)
    xx = xx[mask].reshape((-1, 1))
    yy = yy[mask].reshape((-1, 1))
    zz = zz[mask].reshape((-1, 1))

    lat, lon = eastnorth_to_latlon(xx, yy, zone_number, hemisphere)

    latlonalt = np.hstack((lat, lon, zz))

    return latlonalt


def np2dsm(np):
    pass


if __name__ == '__main__':
    pass
