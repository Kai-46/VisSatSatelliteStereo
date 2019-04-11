from lib.dsm_util import read_dsm_tif
from visualization.plot_height_map import plot_height_map


def plot_dsm_tif(tif_file, out_file, maskout=None, save_cbar=False, force_range=None):
    dsm, meta_dict = read_dsm_tif(tif_file)
    plot_height_map(dsm, out_file, maskout=maskout, save_cbar=save_cbar, force_range=force_range)
