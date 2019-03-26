from lib.image_util import read_image, parse_proj_str
import numpy as np
import utm


# each pixel of a dsm is (east, north, height)
# return (east, north, height) and (latitude, longitude, height)
def dsm2np(dsm_file):
    dsm, geo, proj, meta, width, height = read_image(dsm_file)
    zone_number, hemisphere = parse_proj_str(proj)
    northern = True if hemisphere == 'N' else False
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
    nan_value = -9999.0
    mask = zz > nan_value
    xx = xx[mask].reshape((-1, 1))
    yy = yy[mask].reshape((-1, 1))
    zz = zz[mask].reshape((-1, 1))

    utm_pts = np.hstack((xx, yy, zz))
    latlon_pts = np.copy(utm_pts)
    for i in range(latlon_pts.shape[0]):
        lat, lon = utm.to_latlon(utm_pts[i, 0], utm_pts[i, 1], zone_number, northern=northern)
        latlon_pts[i, 0] = lat
        latlon_pts[i, 1] = lon
    return utm_pts, latlon_pts


def np2dsm(np):
    pass


if __name__ == '__main__':
    tif = '/data2/kz298/mvs3dm_result/MasterSequesteredPark/evaluation/eval_ground_truth.tif'

    utm_pts, latlon_pts = dsm2np(tif)

    np.save('/data2/kz298/mvs3dm_result/MasterSequesteredPark/ground_truth/ground_truth_utm.npy', utm_pts)
    np.save('/data2/kz298/mvs3dm_result/MasterSequesteredPark/ground_truth/ground_truth_latlon.npy', latlon_pts)

    print('hello')
