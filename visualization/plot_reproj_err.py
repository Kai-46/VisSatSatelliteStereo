import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np


# reproj_errs should be a numpy array
def plot_reproj_err(reproj_errs, fpath):
    plt.figure(figsize=(14, 5), dpi=80)
    min_val = np.min(reproj_errs)
    mean_val = np.mean(reproj_errs)
    median_val = np.median(reproj_errs)
    max_val = np.max(reproj_errs)

    # clip reproj_errs at 4 pixels for better visualization
    reproj_errs = np.clip(reproj_errs, 0.0, 4.0)

    plt.hist(reproj_errs, bins=50, density=True, cumulative=False)
    max_points_err = np.max(reproj_errs)
    plt.xticks(np.arange(0, max_points_err + 0.01, 0.1))
    plt.yticks(np.arange(0, 1.01, 0.1))
    plt.xlabel('reprojection error (# pixels)')
    plt.ylabel('pdf')
    plt.title(
        'total # of sparse 3D points: {}\nreproj. err. (pixels): min {:.6f}, mean {:.6f}, median {:.6f}, max {:.6f}'
        .format(reproj_errs.size, min_val, mean_val,
                median_val, max_val))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(fpath)
    plt.close('all')
