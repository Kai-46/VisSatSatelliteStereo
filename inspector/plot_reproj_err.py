import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot_reproj_err(reproj_errs, fpath):
    plt.figure(figsize=(14, 5), dpi=80)
    plt.hist(reproj_errs, bins=20, density=True, cumulative=True)
    max_points_err = max(reproj_errs)
    plt.xticks(np.arange(0, max_points_err + 0.01, 0.1))
    plt.yticks(np.arange(0, 1.01, 0.1))
    plt.xlabel('reprojection error (# pixels)')
    plt.ylabel('cdf')
    plt.title(
        'total # of sparse 3D points: {}\nreproj. err. (pixels): min {:.6f}, mean {:.6f}, median {:.6f}, max {:.6f}'
        .format(len(reproj_errs), min(reproj_errs), np.mean(reproj_errs),
                np.median(reproj_errs), max_points_err))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(fpath)
    plt.close('all')
