import utm
import numpy as np
# from .parallel_apply_along_axis import parallel_apply_along_axis
# import functools
import pyproj

# pyproj implementation of the coordinate conversion
def latlon_to_eastnorh(lat, lon):
    # assume all the points are either on north or south hemisphere
    assert(np.all(lat >= 0) or np.all(lat < 0))

    if lat[0, 0] >= 0: # north hemisphere
        south = False
    else:
        south = True

    _, _, zone_number, _ = utm.from_latlon(lat[0, 0], lon[0, 0])

    proj = pyproj.Proj(proj='utm', ellps='WGS84', zone=zone_number, south=south)
    east, north = proj(lon, lat)
    return east, north

def eastnorth_to_latlon(east, north, zone_number, hemisphere):
    if hemisphere == 'N':
        south = False
    else:
        south = True

    proj = pyproj.Proj(proj='utm', ellps='WGS84', zone=zone_number, south=south)
    lon, lat = proj(east, north, inverse=True)
    return lat, lon


# def map_to_utm(latlon):
#     tmp = utm.from_latlon(latlon[0], latlon[1])
#
#     return np.array([tmp[0], tmp[1]])
#
#
# def latlon_to_eastnorh(lat, lon):
#     latlon = np.concatenate((lat, lon), axis=1)
#
#     tmp = np.apply_along_axis(map_to_utm, 1, latlon)
#     # tmp = parallel_apply_along_axis(map_to_utm, 1, latlon)
#     east = tmp[:, 0:1]
#     north = tmp[:, 1:2]
#
#     return east, north
#
#
# def map_to_latlon(eastnorth, zone_number, northern):
#     tmp = utm.to_latlon(eastnorth[0], eastnorth[1], zone_number, northern=northern)
#
#     return np.array([tmp[0], tmp[1]])
#
#
# def eastnorth_to_latlon(east, north, zone_number, hemisphere):
#     eastnorth = np.concatenate((east, north), axis=1)
#
#     northern = True if hemisphere == 'N' else False
#     map_to_latlon_partial = functools.partial(map_to_latlon, zone_number=zone_number, northern=northern)
#     tmp = np.apply_along_axis(map_to_latlon_partial, 1, eastnorth)
#
#     # tmp = parallel_apply_along_axis(map_to_latlon_partial, 1, eastnorth)
#     lat = tmp[:, 0:1]
#     lon = tmp[:, 1:2]
#
#     return lat, lon


# if __name__ == '__main__':
#     # a point on north hemisphere
#     lat = np.array([47.9941214]).reshape((1, 1))
#     lon = np.array([7.8509671]).reshape((1, 1))
#     print('lat, lon: {}'.format(np.hstack((lat, lon))))
#
#     east, north = latlon_to_eastnorh(lat, lon)
#     print('previous utm: {}'.format(np.hstack((east, north))))
#
#     east, north = pyproj_latlon_to_eastnorh(lat, lon)
#     print('current utm: {}'.format(np.hstack((east, north))))
#
#     lat, lon = pyproj_eastnorth_to_latlon(east, north, zone_number=32, hemisphere='N')
#     print('lat, lon: {}'.format(np.hstack((lat, lon))))
#
#     print('\n')
#     lat = np.array([-47.9941214]).reshape((1, 1))
#     lon = np.array([7.8509671]).reshape((1, 1))
#     print('lat, lon: {}'.format(np.hstack((lat, lon))))
#
#     east, north = latlon_to_eastnorh(lat, lon)
#     print('previous utm: {}'.format(np.hstack((east, north))))
#
#     east, north = pyproj_latlon_to_eastnorh(lat, lon)
#     print('current utm: {}'.format(np.hstack((east, north))))
#
#     lat, lon = pyproj_eastnorth_to_latlon(east, north, zone_number=32, hemisphere='S')
#     print('lat, lon: {}'.format(np.hstack((lat, lon))))