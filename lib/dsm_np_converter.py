import numpy as np
from lib.dsm_util import read_dsm_tif


# each pixel of a dsm is (east, north, alt)
# return (east, north, alt)
def dsm2np(dsm_file):
    dsm, meta_dict = read_dsm_tif(dsm_file)

    zz = dsm.reshape((-1, 1))
    valid_mask = np.logical_not(np.isnan(zz))
    zz = zz[valid_mask].reshape((-1, 1))

    # utm points
    xx = np.linspace(meta_dict['ul_northing'], meta_dict['lr_northing'], meta_dict['img_height'])
    yy = np.linspace(meta_dict['ul_easting'], meta_dict['lr_easting'], meta_dict['img_width'])
    xx, yy = np.meshgrid(xx, yy, indexing='ij')     # xx, yy are of shape (height, width)
    xx = xx.reshape((-1, 1))
    yy = yy.reshape((-1, 1))

    xx = xx[valid_mask].reshape((-1, 1))
    yy = yy[valid_mask].reshape((-1, 1))

    return np.hstack((yy, xx, zz))


def np2dsm(np):
    pass


if __name__ == '__main__':
    pass