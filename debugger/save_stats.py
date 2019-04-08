import numpy as np
from lib.save_image_only import save_image_only
import os


def save_stats(stats, name, out_dir):
    np.save(os.path.join(out_dir, '{}.npy'.format(name)), stats)

    nan_mask = np.isnan(stats)
    stats[nan_mask] = np.nanmin(stats)
    save_image_only(stats, os.path.join(out_dir, '{}.jpg'.format(name)), save_cbar=True)
    save_image_only(1.0-np.float32(nan_mask), os.path.join(out_dir, '{}.mask.jpg'.format(name)), plot=False)
