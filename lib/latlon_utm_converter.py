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

import utm
import numpy as np
from .parallel_apply_along_axis import parallel_apply_along_axis
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
