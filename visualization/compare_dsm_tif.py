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
    # plot_height_map(dsm_1, os.path.join(out_dir, '{}.jpg'.format(dsm_file_1_name)),
    #                 maskout=dsm_0_nan_mask, force_range=(min_val, max_val),
    #                 save_cbar=True)

    plot_height_map(dsm_1, os.path.join(out_dir, '{}.jpg'.format(dsm_file_1_name)),
                    maskout=dsm_0_nan_mask,
                    save_cbar=True)

    signed_err = dsm_1 - dsm_0

    plot_error_map(signed_err, os.path.join(out_dir, '{}.error.jpg'.format(dsm_file_1_name)), maskout=dsm_0_nan_mask,
                   force_range=(-1.5, 1.5), interval=0.1)


if __name__ == '__main__':
    work_dir = '/data2/kz298/mvs3dm_result/MasterProvisional2'
    gt_dsm = os.path.join(work_dir, 'evaluation/eval_ground_truth.tif')
    test_dsm = os.path.join(work_dir, 'mvs_results/aggregate_2p5d/aggregate_2p5d.tif')
    out_dir = os.path.join(work_dir, 'mvs_results/aggregate_2p5d/evaluation')

    compare(gt_dsm, test_dsm, out_dir, align=True)
