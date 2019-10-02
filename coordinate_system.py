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

import os
import numpy as np
import json
from lib.latlonalt_enu_converter import latlonalt_to_enu, enu_to_latlonalt


# global: (xx, yy, zz) = (lat, lon, alt)
# local: enu

def local_to_global(work_dir, xx, yy, zz):
    with open(os.path.join(work_dir, 'aoi.json')) as fp:
        bbx = json.load(fp)
    
    lat0 = (bbx['lat_min'] + bbx['lat_max']) / 2.0
    lon0 = (bbx['lon_min'] + bbx['lon_max']) / 2.0
    alt0 = bbx['alt_min']

#     with open(os.path.join(work_dir, 'geo_grid.json')) as fp:
#         geo_grid = json.load(fp)
#     lat0 = geo_grid['observer_lat']
#     lon0 = geo_grid['observer_lon']
#     alt0 = geo_grid['observer_alt']

    xx, yy, zz = enu_to_latlonalt(xx, yy, zz, lat0, lon0, alt0)

    return xx, yy, zz


def global_to_local(work_dir, xx, yy, zz):
    with open(os.path.join(work_dir, 'aoi.json')) as fp:
        bbx = json.load(fp)

    lat0 = (bbx['lat_min'] + bbx['lat_max']) / 2.0
    lon0 = (bbx['lon_min'] + bbx['lon_max']) / 2.0
    alt0 = bbx['alt_min']

#     with open(os.path.join(work_dir, 'geo_grid.json')) as fp:
#         geo_grid = json.load(fp)
#     lat0 = geo_grid['observer_lat']
#     lon0 = geo_grid['observer_lon']
#     alt0 = geo_grid['observer_alt']

    xx, yy, zz = latlonalt_to_enu(xx, yy, zz, lat0, lon0, alt0)

    return xx, yy, zz
