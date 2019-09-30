import numpy as np
from visualization.save_image_only import save_image_only
from matplotlib.colors import ListedColormap


def plot_height_map(height_map, out_file, maskout=None, save_cbar=False, force_range=None):
    height_map = height_map.copy()

    if force_range is None:
        min_val, max_val = np.nanpercentile(height_map, [1, 99])
        force_range = (min_val, max_val)

    min_val, max_val = force_range
    height_map = np.clip(height_map, min_val, max_val)
    height_map[0, 0] = min_val
    height_map[0, 1] = max_val

    cmap_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'colormap_height.txt')
    colors = np.loadtxt(cmap_file)
    colors = (np.float32(colors) / 255.0).tolist()
    cmap = ListedColormap(colors)

    # save image and mask
    save_image_only(height_map, out_file, cmap=cmap, save_cbar=save_cbar, maskout=maskout, save_mask=True, plot=True)
