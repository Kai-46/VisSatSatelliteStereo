#  ===============================================================================================================
#  Copyright (c) 2019, Cornell University. All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without modification, are permitted provided that
#  the following conditions are met:
#
#      * Redistributions of source code must retain the above copyright otice, this list of conditions and
#        the following disclaimer.
#
#      * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and
#        the following disclaimer in the documentation and/or other materials provided with the distribution.
#
#      * Neither the name of Cornell University nor the names of its contributors may be used to endorse or
#        promote products derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
#  WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
#  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE
#  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
#  TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
#  HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
#   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
#  OF SUCH DAMAGE.
#
#  Author: Kai Zhang (kz298@cornell.edu)
#
#  The research is based upon work supported by the Office of the Director of National Intelligence (ODNI),
#  Intelligence Advanced Research Projects Activity (IARPA), via DOI/IBC Contract Number D17PC00287.
#  The U.S. Government is authorized to reproduce and distribute copies of this work for Governmental purposes.
#  ===============================================================================================================


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
