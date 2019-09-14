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

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np


# reproj_errs should be a numpy array
def plot_reproj_err(reproj_errs, fpath):
    plt.figure(figsize=(14, 5), dpi=80)
    plt.hist(reproj_errs, bins=50, density=True, cumulative=False)
    max_points_err = np.max(reproj_errs)
    plt.xticks(np.arange(0, max_points_err + 0.01, 0.1))
    plt.yticks(np.arange(0, 1.01, 0.1))
    plt.xlabel('reprojection error (# pixels)')
    plt.ylabel('pdf')
    plt.title(
        'total # of sparse 3D points: {}\nreproj. err. (pixels): min {:.6f}, mean {:.6f}, median {:.6f}, max {:.6f}'
        .format(reproj_errs.size, np.min(reproj_errs), np.mean(reproj_errs),
                np.median(reproj_errs), max_points_err))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(fpath)
    plt.close('all')
