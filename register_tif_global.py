import numpy as np
from visualization.plot_error_map import plot_error_map
from visualization.plot_height_map import plot_height_map
from visualization.plot_error_dist import plot_err_dist
from lib.dsm_util import read_dsm_tif
import os
import cv2
import multiprocessing


def split_big_list(big_list, num_small_lists):
    cnt = len(big_list)
    indices = np.array(np.arange(cnt, dtype=np.int32))
    indices = np.array_split(indices, num_small_lists)

    small_lists = []
    for i in range(num_small_lists):
        subindices = indices[i]
        if subindices.size > 0:
            idx1 = subindices[0]
            idx2 = subindices[-1]
            small_lists.append(big_list[idx1:idx2+1])

    return small_lists


def calc_score(source, target):
    abs_err = np.abs(source - target)
    median_err = np.nanmedian(abs_err)

    completeness_threshold = 1.0
    valid_target_cell_cnt = np.sum(np.logical_not(np.isnan(target)))
    completeness = np.sum(np.less(abs_err, completeness_threshold)) / valid_target_cell_cnt

    valid_abs_err_cell_cnt = np.sum(np.logical_not(np.isnan(abs_err)))
    rmse = np.sqrt(np.nansum(abs_err * abs_err) / valid_abs_err_cell_cnt)

    return median_err, completeness, rmse


def align_worker(return_dict, proc_idx, source, target, target_pad_width, xy_candi_shifts):
    target = target[target_pad_width:-target_pad_width, target_pad_width:-target_pad_width]
    h, w = target.shape[:2]

    best_dx = None
    best_dy = None
    best_dz = None
    best_median_err = np.inf

    for xy_shift in xy_candi_shifts:
        dx, dy = xy_shift

        ul_x = target_pad_width + dx
        ul_y = target_pad_width + dy
        source_shifted = source[ul_y:ul_y+h, ul_x:ul_x+w]

        # shift source in z direction and compute median error
        median_dz = np.nanmedian(target - source_shifted)
        source_shifted = source_shifted + median_dz
        median_err = np.nanmedian(np.abs(source_shifted - target))

        if median_err < best_median_err:
            best_median_err = median_err
            best_dx = dx
            best_dy = dy
            best_dz = median_dz

    return_dict[proc_idx] = (best_dx, best_dy, best_dz, best_median_err)


# align source_height_map to target_height_map
# source, target are of the same size
def align(source, target, out_dir, target_pad_width, search_radius, cell_physical_size):
    plot_height_map(source, os.path.join(out_dir, 'source_before_align.jpg'), save_cbar=True)
    plot_height_map(target, os.path.join(out_dir, 'target_before_align.jpg'), save_cbar=True)

    # all candidate shifts
    xy_candi_shifts = []
    for dx in range(-search_radius, search_radius+1, 1):
        for dy in range(-search_radius, search_radius+1, 1):
            xy_candi_shifts.append((dx, dy))

    # divide the candidate shits into subset
    max_processes = 8
    xy_candi_shifts = split_big_list(xy_candi_shifts, max_processes)
    num_processes = len(xy_candi_shifts)    # actual number of processes to be launched

    # shared variable to store results returned by multiple processes
    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    jobs = []
    for i in range(num_processes):
        p = multiprocessing.Process(target=align_worker, args=(return_dict, i, source, target, target_pad_width, xy_candi_shifts[i]))
        jobs.append(p)
        p.start()

    for proc in jobs:
        proc.join()
    
    results = return_dict.values()
    # print('{}'.format(results))
    results = sorted(results, key=lambda result: result[3])     # sort by median_err
    best_dx, best_dy, best_dz, best_median_err = results[0]

    # remove padded margin for target
    target = target[target_pad_width:-target_pad_width, target_pad_width:-target_pad_width]
    h, w = target.shape[:2]

    # before alignment
    source_shifted = source[target_pad_width:-target_pad_width, target_pad_width:-target_pad_width]
    median_err, completeness, rmse = calc_score(source_shifted, target)
    print('before alignment, median err: {}, completeness: {}, rmse: {}'.format(median_err, completeness, rmse))

    # after alignment, apply shifts
    ul_x = target_pad_width + best_dx
    ul_y = target_pad_width + best_dy
    source_shifted = source[ul_y:ul_y+h, ul_x:ul_x+w] + best_dz
    median_err, completeness, rmse = calc_score(source_shifted, target)
    print('best dx, dy, dz: {}, {}, {}, best median err: {}, completeness: {}, rmse: {}'.format(
        best_dx, best_dy, best_dz, median_err, completeness, rmse))

    with open(os.path.join(out_dir, 'offset.txt'), 'w') as fp:
        cell_w, cell_h = cell_physical_size
        fp.write('cell_width (meters): {}\n'.format(cell_w))
        fp.write('cell_height (meters): {}\n'.format(cell_h))
        fp.write('best_dx (cells): {}\n'.format(best_dx))
        fp.write('best_dy (cells): {}\n'.format(best_dy))
        fp.write('best_dz (meters): {}\n'.format(best_dz))
        fp.write('best_median_err (meters): {}\n'.format(median_err))
        fp.write('completeness (<1 meter): {}\n'.format(completeness))
        fp.write('rmse (meters): {}\n'.format(rmse))

    err = source_shifted - target
    mask = np.logical_not(np.isnan(err))
    plot_err_dist(np.abs(err[mask]), os.path.join(out_dir, 'error_dist.jpg'))

    min_val, max_val = np.nanpercentile(target, (1, 99))
    plot_height_map(target, os.path.join(out_dir, 'target_after_align.jpg'), save_cbar=True)
    plot_height_map(source_shifted, os.path.join(out_dir, 'source_after_align.jpg'), maskout=np.isnan(target),
                    save_cbar=True, force_range=(min_val, max_val))
    plot_error_map(err, os.path.join(out_dir, 'error_map.jpg'), force_range=(-2.0, 2.0))


def register(source_tif, target_tif, out_dir):
    print('evaluating {}...'.format(source_tif))

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    source, source_meta = read_dsm_tif(source_tif)
    target, target_meta = read_dsm_tif(target_tif)

    # assert source and target are at the same resolution
    assert (np.abs(source_meta['east_resolution'] - target_meta['east_resolution']) < 1e-5 and
            np.abs(source_meta['north_resolution'] - target_meta['north_resolution'] < 1e-5))

    cell_physical_size = (target_meta['east_resolution'], target_meta['north_resolution'])

    search_radius = 30  # default is 15 meters search radius 
    target_pad_width = search_radius + 10
    target = np.pad(target, ((target_pad_width, target_pad_width), (target_pad_width, target_pad_width)), mode='constant', constant_values=np.nan)

    ul_x = (target_meta['ul_easting'] - source_meta['ul_easting']) / source_meta['east_resolution']
    ul_y = (source_meta['ul_northing'] - target_meta['ul_northing']) / source_meta['north_resolution']
    print('ul_x, ul_y: {}, {}'.format(ul_x, ul_y))
    ul_x -= target_pad_width
    ul_y -= target_pad_width
    affine_mat = np.array([[1.0, 0.0, -ul_x], [0.0, 1.0, -ul_y]])
    source_crop = cv2.warpAffine(source, affine_mat, (target.shape[1], target.shape[0]), borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)

    align(source_crop, target, out_dir, target_pad_width, search_radius, cell_physical_size)


if __name__ == '__main__':
    # usage: python3 {tif to be evaluated} {ground-truth tif} {output directory}
    import sys
    tif_to_eval = sys.argv[1]
    tif_gt = sys.argv[2]
    out_dir = sys.argv[3]
    register(tif_to_eval, tif_gt, out_dir)
