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