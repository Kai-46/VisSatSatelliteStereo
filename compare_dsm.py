from lib.image_util import read_image
from lib.save_image_only import save_image_only
import numpy as np
from debugger.signed_colormap import get_signed_colormap


def plot_tif(tif_file, out_file, maskout=None):
    data, geo, proj, meta, width, height = read_image(tif_file)
    data = data[:, :, 0]
    nan_mask = data < -9998  # actually -9999
    if maskout is not None:
        nan_mask = np.logical_or(nan_mask, maskout)
    data[nan_mask] = np.min(data[np.logical_not(nan_mask)])
    save_image_only(data, out_file, save_cbar=True)
    save_image_only(1.0-np.float32(nan_mask), out_file + '.mask.jpg', plot=False)


def compare(dsm_file_0, dsm_file_1, out_file):
    dsm_0, geo, proj, meta, width, height = read_image(dsm_file_0)
    dsm_0 = dsm_0[:, :, 0]
    dsm_0_nan_mask = dsm_0 < -9998  # actually -9999
    dsm_0[dsm_0_nan_mask] = np.nan

    dsm_1, geo, proj, meta, width, height = read_image(dsm_file_1)
    dsm_1 = dsm_1[:, :, 0]
    dsm_1_nan_mask = dsm_1 < -9998
    dsm_1[dsm_1_nan_mask] = np.nan

    signed_err = dsm_1 - dsm_0

    # visualize
    clip_min = -1.0
    clip_max = 1.0
    signed_err[signed_err < clip_min] = clip_min
    signed_err[signed_err > clip_max] = clip_max
    signed_err[dsm_0_nan_mask] = 0.0
    cmap, norm = get_signed_colormap(vmin=clip_min, vmax=clip_max)

    nan_mask = np.isnan(signed_err)
    signed_err[nan_mask] = np.nanmin(signed_err)
    save_image_only(signed_err, out_file, save_cbar=True, cmap=cmap, norm=norm)
    save_image_only(1.0-np.float32(nan_mask), out_file + '.mask.jpg', plot=False)

    return dsm_0_nan_mask


if __name__ == '__main__':
    dsm_file_0 = '/data2/kz298/mvs3dm_result/MasterProvisional2/evaluation/eval_ground_truth.tif'
    #dsm_file_1 = '/data2/kz298/mvs3dm_result/MasterProvisional2/mvs_results/merged_dsm_aligned.tif'
    dsm_file_1 = '/data2/kz298/mvs3dm_result/MasterProvisional2/mvs_results/colmap_fused_dsm_aligned.tif'

    out_file = dsm_file_1 + '.error.jpg'
    dsm_file_0_nan_mask = compare(dsm_file_0, dsm_file_1, out_file)
    plot_tif(dsm_file_1, dsm_file_1[:-4] + '.jpg', maskout=dsm_file_0_nan_mask)
