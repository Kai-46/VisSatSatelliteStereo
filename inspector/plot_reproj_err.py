import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# reproj_errs should be a numpy array
def plot_reproj_err(reproj_errs, fpath):
    plt.figure(figsize=(14, 5), dpi=80)
    # print('what the heck?...')
    # print(reproj_errs.size)
    # #plt.hist(reproj_errs, bins=50)
    # print('omg!')
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
