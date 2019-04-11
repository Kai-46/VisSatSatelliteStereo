import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm


def get_signed_colormap(vmin=-2.0, vmax=2.0, interval=0.1):
    # define the colormap
    cmap = plt.get_cmap('PuOr')

    # extract all colors from the .jet map
    cmaplist = [cmap(i) for i in range(cmap.N)]
    # create the new map
    cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)

    # define the bins and normalize and forcing 0 to be part of the colorbar!
    eps = 1e-10
    bounds = np.arange(vmin-eps, vmax+eps, interval)
    idx = np.searchsorted(bounds, 0)
    bounds=np.insert(bounds, idx, 0)
    norm = BoundaryNorm(bounds, cmap.N)

    return cmap, norm
