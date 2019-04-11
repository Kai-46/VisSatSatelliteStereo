import numpy as np
from visualization.save_image_only import save_image_only


def plot_height_map(height_map, out_file, maskout=None, save_cbar=False, force_range=None):
    height_map = height_map.copy()
    if force_range is not None:
        min_val, max_val = force_range
        height_map = np.clip(height_map, min_val, max_val)
        height_map[0, 0] = min_val
        height_map[0, 1] = max_val

    # save image and mask
    save_image_only(height_map, out_file, cmap='jet', save_cbar=save_cbar, maskout=maskout, save_mask=True, plot=True)
