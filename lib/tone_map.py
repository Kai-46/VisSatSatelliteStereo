import numpy as np
import imageio
import os


# hdr_img is 16-bit, while ldr_img is 8 bit
def tone_map(hdr_img, ldr_img):
    # scale to [0, 255]
    im = imageio.imread(hdr_img).astype(dtype=np.float64)

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
    if os.path.exists(ldr_img):
        os.remove(ldr_img)

    imageio.imwrite(ldr_img, im.astype(dtype=np.uint8))