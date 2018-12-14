# return intersection of two bounding boxes, and overlap ratio for each bbx

# each bbx is specified as: (ul_col, ul_row, width, height)

def check_bbx(first_bbx, second_bbx):
    col_idxs = [(1, first_bbx[0]),
                (1, first_bbx[0] + first_bbx[2] - 1),
                (2, second_bbx[0]),
                (2, second_bbx[0] + second_bbx[2] - 1)]
    row_idxs = [(1, first_bbx[1]),
                (1, first_bbx[1] + first_bbx[3] - 1),
                (2, second_bbx[1]),
                (2, second_bbx[1] + second_bbx[3] - 1)]

    # sort row_idxs and col_idxs
    col_idxs = sorted(col_idxs, key=lambda x: x[1])
    row_idxs = sorted(row_idxs, key=lambda x: x[1])

    # no intersection
    flag1 = col_idxs[1][1] == col_idxs[2][1] or row_idxs[1][1] == row_idxs[2][1]
    flag2 = col_idxs[0][0] == col_idxs[1][0] or row_idxs[0][0] == row_idxs[1][0]
    if flag1 or flag2:
        intersect = None
        first_overlap = 0.
        second_overlap = 0.
    else:
        # get intersect
        intersect_w = col_idxs[2][1] - col_idxs[1][1] + 1
        intersect_h = row_idxs[2][1] - row_idxs[1][1] + 1
        intersect = [col_idxs[1][1], row_idxs[1][1], intersect_w, intersect_h]

        first_overlap = float(intersect_w * intersect_h) / (first_bbx[2] * first_bbx[3])
        second_overlap = float(intersect_w * intersect_h) / (second_bbx[2] * second_bbx[3])

    return intersect, first_overlap, second_overlap


if __name__ == '__main__':
    first_bbx = (0, 0, 50, 40)
    second_bbx = (48, 10, 30, 40)

    print(check_bbx(first_bbx, second_bbx))
    print(check_bbx(second_bbx, first_bbx))