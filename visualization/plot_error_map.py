from visualization.signed_colormap import get_signed_colormap
from visualization.save_image_only import save_image_only
import numpy as np


def plot_error_map(error_map, out_file, maskout=None, force_range=None, interval=None):
    error_map = error_map.copy()
    if force_range is not None:
        min_val, max_val = force_range
        error_map = np.clip(error_map, min_val, max_val)
        error_map[0, 0] = min_val
        error_map[0, 1] = max_val
    else:
        min_val = np.nanmin(error_map)
        max_val = np.nanmax(error_map)

    if interval is None:
        interval = (max_val - min_val) / 10.0

    cmap, norm = get_signed_colormap(vmin=min_val, vmax=max_val, interval=interval)

    # save image and mask
    save_image_only(error_map, out_file, maskout=maskout, cmap=cmap, norm=norm, save_cbar=True, plot=True)

    #save_image_only(error_map, out_file, maskout=maskout, cmap='jet', save_cbar=True, plot=True)
