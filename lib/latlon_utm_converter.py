import utm
import numpy as np
from lib.parallel_apply_along_axis import parallel_apply_along_axis
import functools


def map_to_utm(latlon):
    tmp = utm.from_latlon(latlon[0], latlon[1])

    return np.array([tmp[0], tmp[1]])


def latlon_to_eastnorh(lat, lon):
    latlon = np.concatenate((lat, lon), axis=1)

    # tmp = np.apply_along_axis(map_to_utm, 1, latlonalt)
    tmp = parallel_apply_along_axis(map_to_utm, 1, latlon)
    east = tmp[:, 0:1]
    north = tmp[:, 1:2]

    return east, north


def map_to_latlon(eastnorth, zone_number, northern):
    tmp = utm.to_latlon(eastnorth[0], eastnorth[1], zone_number, northern=northern)

    return np.array([tmp[0], tmp[1]])


def eastnorth_to_latlon(east, north, zone_number, hemisphere):
    eastnorth = np.concatenate((east, north), axis=1)

    northern = True if hemisphere == 'N' else False
    map_to_latlon_partial = functools.partial(map_to_latlon, zone_number=zone_number, northern=northern)
    tmp = parallel_apply_along_axis(map_to_latlon_partial, 1, eastnorth)
    lat = tmp[:, 0:1]
    lon = tmp[:, 1:2]

    return lat, lon
