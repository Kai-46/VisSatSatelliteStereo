import cv2
import numpy as np

# img_src is numpy array, affine matrix is 2*3 matrix
# image index is col, row
# keep all pixels in the source image
# return img_dst, off_set
def warp_affine(img_src, affine_matrix):
    height, width = img_src.shape

    # compute bounding box
    bbx = np.dot(affine_matrix, np.array([[0, width, width, 0],
                                          [0, 0, height, height],
                                          [1, 1, 1, 1]]))
    col_min = np.min(bbx[0, :])
    col_max = np.max(bbx[0, :])
    row_min = np.min(bbx[1, :])
    row_max = np.max(bbx[1, :])

    off_set = (col_min, row_min)
    w = int(np.round(col_max - col_min + 1))
    h = int(np.round(row_max - row_min + 1))

    # add offset to the affine_matrix
    affine_matrix[0, 2] -= col_min
    affine_matrix[1, 2] -= row_min

    img_dst = cv2.warpAffine(img_src, affine_matrix, (w, h))

    return img_dst, off_set, (w, h)


if __name__ == '__main__':
    import imageio

    img_src = imageio.imread('/data2/kz298/core3d_result/aoi-d1-wpafb/images/000_20141026163007.jpg').astype(dtype=np.float64)

    affine_matrix = np.array([[1, -0.5, 0],
                              [0, 1, 0]])
    img_dst, off_set = warp_affine(img_src, affine_matrix)

    imageio.imwrite('/data2/kz298/test.jpg', img_dst.astype(np.uint8))