import numpy as np
import imageio
import os


# in_png is 16-bit png, while out_png is 8 bit
def tone_map(in_png, out_png):
    # scale to [0, 255]
    im = imageio.imread(in_png).astype(dtype=np.float64)

    # tmp = im.reshape((-1, 1))
    # tmp = (tmp - np.min(tmp)) / (np.max(tmp) - np.min(tmp))
    # plt.hist(tmp, bins=100, density=True, cumulative=True)
    # plt.xlabel('normalized intensity')
    # plt.title('original image intensity')
    # plt.show()

    im = np.power(im, 1.0 / 2.2)  # non-uniform sampling

    # tmp = im.reshape((-1, 1))
    # tmp = (tmp - np.min(tmp)) / (np.max(tmp) - np.min(tmp))
    # plt.hist(tmp, bins=100, density=True, cumulative=True)
    # plt.xlabel('normalized intensity')
    # plt.title('after applying the non-linear encoding')
    # plt.show()

    # cut off the small values
    below_thres = np.percentile(im.reshape((-1, 1)), 1)
    im[im < below_thres] = below_thres
    # cut off the big values
    above_thres = np.percentile(im.reshape((-1, 1)), 99.5)
    im[im > above_thres] = above_thres
    im = 255 * (im - below_thres) / (above_thres - below_thres)

    # tmp = im.reshape((-1, 1))
    # tmp = (tmp - np.min(tmp)) / (np.max(tmp) - np.min(tmp))
    # plt.hist(tmp, bins=100, density=True, cumulative=True)
    # plt.xlabel('normalized intensity')
    # plt.title('after applying the non-linear encoding')
    # plt.show()

    # remove the unneeded one
    if os.path.exists(out_png):
        os.remove(out_png)

    imageio.imwrite(out_png, im.astype(dtype=np.uint8))