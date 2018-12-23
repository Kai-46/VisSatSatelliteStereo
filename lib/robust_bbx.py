import numpy as np


def robust_bbx(col, row):
    col_min, col_max = np.percentile(col, [5, 95])
    row_min, row_max = np.percentile(row, [5, 95])

    return int(col_min), int(row_min), int(col_max - col_min + 1), int(row_max - row_min + 1)