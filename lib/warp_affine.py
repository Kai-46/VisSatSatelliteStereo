import cv2
import numpy as np

# img_src is numpy array, affine matrix is 2*3 matrix
# image index is col, row
# keep all pixels in the source image
# return img_dst, off_set
def warp_affine(img_src, affine_matrix, no_blank_margin=True):
    height, width = img_src.shape[:2]

    # compute bounding box
    bbx = np.dot(affine_matrix, np.array([[0, width, width, 0],
                                          [0, 0, height, height],
                                          [1, 1, 1, 1]]))

    if no_blank_margin:
        col = np.sort(bbx[0, :])
        row = np.sort(bbx[1, :])

        # leave some small margin
        col_min = int(col[1]) + 3
        row_min = int(row[1]) + 3
        w = int(col[2]) - col_min - 3
        h = int(row[2]) - row_min - 3
    else:
        col_min = np.min(bbx[0, :])
        col_max = np.max(bbx[0, :])
        row_min = np.min(bbx[1, :])
        row_max = np.max(bbx[1, :])

        w = int(np.round(col_max - col_min + 1))
        h = int(np.round(row_max - row_min + 1))

    # add offset to the affine_matrix
    affine_matrix[0, 2] -= col_min
    affine_matrix[1, 2] -= row_min

    off_set = (-col_min, -row_min)

    # warp image
    img_dst = cv2.warpAffine(img_src, affine_matrix, (w, h))

    assert (h == img_dst.shape[0] and w == img_dst.shape[1])

    return img_dst, off_set, affine_matrix


if __name__ == '__main__':
    import imageio

    img_src = imageio.imread('/data2/kz298/core3d_result/aoi-d1-wpafb/images/000_20141026163007.jpg').astype(dtype=np.float64)

    affine_matrix = np.array([[1, -0.5, 0],
                              [0, 1, 0]])
    img_dst, off_set = warp_affine(img_src, affine_matrix)

    imageio.imwrite('/data2/kz298/test.jpg', img_dst.astype(np.uint8))