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

    # cmap, norm = get_signed_colormap(vmin=min_val, vmax=max_val, interval=interval)

    # save image and mask
    save_image_only(error_map, out_file, maskout=maskout, cmap='bwr', save_cbar=True, plot=True)

    #save_image_only(error_map, out_file, maskout=maskout, cmap='jet', save_cbar=True, plot=True)
