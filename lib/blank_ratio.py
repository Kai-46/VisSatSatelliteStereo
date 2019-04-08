import numpy as np
import imageio


def blank_ratio(img_path, thres=1e-3):
    im = imageio.imread(img_path).astype(dtype=np.float64) / 255.0

    return np.sum(im < thres) / im.size
