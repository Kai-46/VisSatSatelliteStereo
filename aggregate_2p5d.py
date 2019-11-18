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


import numpy as np
import os
from aggregate_2p5d_util import convert_depth_maps
from lib.ply_np_converter import np2ply
import json
import logging
from lib.dsm_util import read_dsm_tif
from produce_dsm import produce_dsm_from_height
import cv2
import imageio


def run_fuse(work_dir, max_processes=-1):
    # first convert depth maps
    dsm_dir = os.path.join(work_dir, 'colmap/mvs/dsm')
    convert_depth_maps(work_dir, dsm_dir, depth_type='geometric', max_processes=max_processes)

    if not os.path.exists(os.path.join(work_dir, 'mvs_results')):
        os.mkdir(os.path.join(work_dir, 'mvs_results'))

    out_dir = os.path.join(work_dir, 'mvs_results/aggregate_2p5d')
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    all_dsm = []
    dsm_tif_dir = os.path.join(dsm_dir, 'dsm_tif')
    for item in sorted(os.listdir(dsm_tif_dir)):
        dsm, _ = read_dsm_tif(os.path.join(dsm_tif_dir, item))
        all_dsm.append(dsm[:, :, np.newaxis])

        logging.info('dsm {} empty ratio: {} '.format(item, np.sum(np.isnan(dsm)) / dsm.size))

    cnt = len(all_dsm)
    all_dsm = np.concatenate(all_dsm, axis=2)

    # reject two measurements
    num_measurements = cnt - np.sum(np.isnan(all_dsm), axis=2, keepdims=True)
    mask = np.tile(num_measurements <= 2, (1, 1, cnt))
    all_dsm[mask] = np.nan

    # reject outliers based on MAD statistics
    all_dsm_median = np.nanmedian(all_dsm, axis=2, keepdims=True)
    all_dsm_mad = np.nanmedian(np.abs(all_dsm - all_dsm_median), axis=2, keepdims=True)
    outlier_mask = np.abs(all_dsm - all_dsm_median) > all_dsm_mad
    all_dsm[outlier_mask] = np.nan
    all_dsm_mean_no_outliers = np.nanmean(all_dsm, axis=2)

    # median filter
    all_dsm_mean_no_outliers = cv2.medianBlur(all_dsm_mean_no_outliers.astype(np.float32), 3)

    # write tif
    tif_to_write = os.path.join(out_dir, 'aggregate_2p5d_dsm.tif')
    jpg_to_write = os.path.join(out_dir, 'aggregate_2p5d_dsm.jpg')
    ul_e, ul_n, e_size, n_size, e_resolution, n_resolution = produce_dsm_from_height(work_dir, all_dsm_mean_no_outliers, tif_to_write, jpg_to_write)

    void_ratio = np.sum(np.isnan(all_dsm_mean_no_outliers)) / all_dsm_mean_no_outliers.size
    logging.info('\n After aggregation, empty ratio: {} '.format(void_ratio))

    # create a colored point cloud
    xx = ul_n - np.arange(n_size) * n_resolution
    yy = ul_e + np.arange(e_size) * e_resolution
    xx, yy = np.meshgrid(xx, yy, indexing='ij')     # xx, yy are of shape (height, width)
    
    xx = xx.reshape((-1, 1))
    yy = yy.reshape((-1, 1))
    zz = all_dsm_mean_no_outliers.reshape((-1, 1))
    color = imageio.imread(jpg_to_write).reshape((-1, 3))

    valid_mask = np.logical_not(np.isnan(zz)).flatten()
    xx = xx[valid_mask, :]
    yy = yy[valid_mask, :]
    zz = zz[valid_mask, :]
    color = color[valid_mask, :]

    utm_points = np.concatenate((yy, xx, zz), axis=1)
    with open(os.path.join(work_dir, 'aoi.json')) as fp:
        aoi_dict = json.load(fp)
    comment_1 = 'projection: UTM {}{}'.format(aoi_dict['zone_number'], aoi_dict['hemisphere'])
    comments = [comment_1,]
    np2ply(utm_points, os.path.join(out_dir, 'aggregate_2p5d.ply'), 
            color=color, comments=comments, use_double=True)


if __name__ == '__main__':
    work_dir = '/data2/kz298/mvs3dm_result/MasterProvisional2'
    run_fuse(work_dir)
