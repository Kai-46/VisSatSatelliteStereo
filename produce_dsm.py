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
import json
import numpy as np
from visualization.plot_height_map import plot_height_map
from lib.dsm_util import write_dsm_tif
from lib.proj_to_grid import proj_to_grid
import cv2

e_resolution = 0.5  # 0.5 meters per pixel
n_resolution = 0.5 

# points is in UTM
def produce_dsm_from_points(work_dir, points, tif_to_write, jpg_to_write=None):
    with open(os.path.join(work_dir, 'aoi.json')) as fp:
        aoi_dict = json.load(fp)

    # write dsm to tif
    ul_e = aoi_dict['ul_easting']
    ul_n = aoi_dict['ul_northing']

    e_size = int(aoi_dict['width'] / e_resolution) + 1
    n_size = int(aoi_dict['height'] / n_resolution) + 1
    dsm = proj_to_grid(points, ul_e, ul_n, e_resolution, n_resolution, e_size, n_size, propagate=True)
    # median filter
    dsm = cv2.medianBlur(dsm.astype(np.float32), 3)
    write_dsm_tif(dsm, tif_to_write, 
                  (ul_e, ul_n, e_resolution, n_resolution), 
                  (aoi_dict['zone_number'], aoi_dict['hemisphere']), nodata_val=-10000)
    # create a preview file
    if jpg_to_write is not None:
        dsm = np.clip(dsm, aoi_dict['alt_min'], aoi_dict['alt_max'])
        plot_height_map(dsm, jpg_to_write, save_cbar=True)

    return (ul_e, ul_n, e_size, n_size, e_resolution, n_resolution)
    
# points is in UTM
def produce_dsm_from_height(work_dir, height, tif_to_write, jpg_to_write=None):
    with open(os.path.join(work_dir, 'aoi.json')) as fp:
        aoi_dict = json.load(fp)

    # write dsm to tif
    ul_e = aoi_dict['ul_easting']
    ul_n = aoi_dict['ul_northing']
    n_size, e_size = height.shape[:2]
    
    write_dsm_tif(height, tif_to_write, 
                  (ul_e, ul_n, e_resolution, n_resolution), 
                  (aoi_dict['zone_number'], aoi_dict['hemisphere']), nodata_val=-10000)
    # create a preview file
    if jpg_to_write is not None:
        height = np.clip(height, aoi_dict['alt_min'], aoi_dict['alt_max'])
        plot_height_map(height, jpg_to_write, save_cbar=True)

    return (ul_e, ul_n, e_size, n_size, e_resolution, n_resolution)

if __name__ == '__main__':
    pass
