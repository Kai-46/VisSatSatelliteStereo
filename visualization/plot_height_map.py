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
