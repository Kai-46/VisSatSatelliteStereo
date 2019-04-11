import os
import numpy as np
import json
from lib.latlonalt_enu_converter import latlonalt_to_enu, enu_to_latlonalt


# global: (xx, yy, zz) = (lat, lon, alt)
# local: enu

def local_to_global(work_dir, xx, yy, zz):
    with open(os.path.join(work_dir, 'aoi.json')) as fp:
        bbx = json.load(fp)
    lat_min = bbx['lat_min']
    lon_min = bbx['lon_min']
    alt_min = bbx['alt_min']
    xx, yy, zz = enu_to_latlonalt(xx, yy, zz, lat_min, lon_min, alt_min)

    return xx, yy, zz


def global_to_local(work_dir, xx, yy, zz):
    with open(os.path.join(work_dir, 'aoi.json')) as fp:
        bbx = json.load(fp)
    lat_min = bbx['lat_min']
    lon_min = bbx['lon_min']
    alt_min = bbx['alt_min']
    xx, yy, zz = latlonalt_to_enu(xx, yy, zz, lat_min, lon_min, alt_min)

    return xx, yy, zz
