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

from lib.dsm_util import read_dsm_tif
import numpy as np
import os
import shutil
from lib.run_cmd import run_cmd
from visualization.plot_height_map import plot_height_map
from visualization.plot_error_map import plot_error_map


def get_filename(path):
    idx1 = path.rfind('/')
    idx2 = path.rfind('.')

    return path[idx1+1:idx2]


def compare(dsm_file_0, dsm_file_1, out_dir, align=False):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # copy files to out_dir
    dsm_file_0_name = get_filename(dsm_file_0)
    dsm_file_1_name = get_filename(dsm_file_1)

    shutil.copyfile(dsm_file_0, os.path.join(out_dir, '{}.tif'.format(dsm_file_0_name)))
    shutil.copyfile(dsm_file_1, os.path.join(out_dir, '{}.tif'.format(dsm_file_1_name)))

    dsm_file_0 = os.path.join(out_dir, '{}.tif'.format(dsm_file_0_name))
    dsm_file_1 = os.path.join(out_dir, '{}.tif'.format(dsm_file_1_name))

    if align:
        dsm_1, meta_dict1 = read_dsm_tif(dsm_file_1)
        plot_height_map(dsm_1, os.path.join(out_dir, '{}.jpg'.format(dsm_file_1_name)),
                        save_cbar=True)

        cmd = '/home/cornell/kz298/pubgeo/build/align3d {} {} maxt=10.0'.format(dsm_file_0, dsm_file_1)
        run_cmd(cmd)

        dsm_file_1 = os.path.join(out_dir, '{}_aligned.tif'.format(dsm_file_1_name))
        dsm_file_1_name = get_filename(dsm_file_1)

    dsm_0, meta_dict0 = read_dsm_tif(dsm_file_0)

    plot_height_map(dsm_0, os.path.join(out_dir, '{}.jpg'.format(dsm_file_0_name)), save_cbar=True)

    dsm_1, meta_dict1 = read_dsm_tif(dsm_file_1)

    min_val = np.nanmin(dsm_0)
    max_val = np.nanmax(dsm_0)
    dsm_0_nan_mask = np.isnan(dsm_0)
    plot_height_map(dsm_1, os.path.join(out_dir, '{}.jpg'.format(dsm_file_1_name)),
                    maskout=dsm_0_nan_mask, force_range=(min_val, max_val),
                    save_cbar=True)

    signed_err = dsm_1 - dsm_0

    print('median signed error: {}'.format(np.nanmedian(signed_err)))
    abs_err = np.abs(signed_err)
    print('median error: {}'.format(np.nanmedian(abs_err)))
    print('completeness: {}'.format(np.sum(abs_err < 1.0) / (dsm_0.size - np.sum(dsm_0_nan_mask))))

    plot_error_map(signed_err, os.path.join(out_dir, '{}.error.jpg'.format(dsm_file_1_name)), maskout=dsm_0_nan_mask,
                   force_range=(-1.5, 1.5), interval=0.1)


if __name__ == '__main__':
    # test_dsm = '/data2/kz298/mvs3dm_result_bak2/MasterProvisional2/mvs_results_all/aggregate_2p5d/evaluation/aggregate_2p5d.tif'
    # gt_dsm = '/data2/kz298/mvs3dm_result_bak2/MasterProvisional2/mvs_results_all/aggregate_2p5d/evaluation/eval_ground_truth.tif'
    # out_dir = '/data2/kz298/mvs3dm_result/MasterProvisional2/mvs_results_all/aggregate_2p5d/evaluation/icp'
    # compare(gt_dsm, test_dsm, out_dir, align=False)

    work_dir = '/data2/kz298/mvs3dm_result/MasterProvisional2/'
    test_dsm = os.path.join(work_dir, 'mvs_results/aggregate_2p5d/aggregate_2p5d.tif')
    out_dir = os.path.join(work_dir, 'mvs_results/aggregate_2p5d/evaluation')
    eval_ground_truth = os.path.join(work_dir, 'evaluation/eval_ground_truth.tif')
    compare(eval_ground_truth, test_dsm, out_dir, align=False)
