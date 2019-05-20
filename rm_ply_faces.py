from lib.ply_np_converter import ply2np,  np2ply
import numpy as np
from visualization.plot_height_map import plot_height_map
from visualization.plot_error_map import plot_error_map


def ply_to_map(in_ply, out_npy):
    xyz = ply2np(in_ply)

    cols = xyz[:, 0]
    rows = xyz[:, 1]
    alts = xyz[:, 2]

    width = int(np.max(cols)) + 1
    height = int(np.max(rows)) + 1

    dsm = np.empty((height, width))
    dsm.fill(np.nan)
    for i in range(alts.shape[0]):
        r = int(rows[i])
        c = int(cols[i])
        z = alts[i]

        dsm[r, c] = z

    np.save(out_npy, dsm)

    # min_val, max_val = np.nanpercentile(dsm, [0.01, 99])
    # dsm = np.clip(dsm, min_val, max_val)
    plot_height_map(dsm, out_npy[:-4] + '.jpg', save_cbar=True)


if __name__ == '__main__':
    # in_ply = '/data2/kz298/mvs3dm_result/s2p_50_pairs_median.ply'
    # xyz, comments = ply2np(in_ply, return_comments=True, only_xyz=True)
    #
    # out_ply = in_ply[:-4] + '_verts.ply'
    # np2ply(xyz, out_ply, comments, text=True)
    #
    # in_ply = '/data2/kz298/mvs3dm_result/gt.ply'
    # xyz, comments = ply2np(in_ply, return_comments=True, only_xyz=True)
    #
    # out_ply = in_ply[:-4] + '_verts.ply'
    # np2ply(xyz, out_ply, comments, text=True)

    # in_ply = '/data2/kz298/mvs3dm_result/s2p_50_pairs_median_verts.ply'
    # ply_to_map(in_ply, in_ply[:-4] + '.npy')
    #
    # in_ply = '/data2/kz298/mvs3dm_result/gt_verts.ply'
    # ply_to_map(in_ply, in_ply[:-4] + '.npy')

    gt_dsm =  np.load('/data2/kz298/mvs3dm_result/s2p_result/gt_verts.npy')
    test_dsm = np.load('/data2/kz298/mvs3dm_result/s2p_result/s2p_50_pairs_median_verts.npy')
    valid_mask = np.logical_not(np.isnan(gt_dsm))
    # for shift in list(np.arange(1.7, 1.8, 0.01)):

    best_median_abs_err = 1e20
    best_completeness = 0.0
    best_shift = 0.0
    for shift in [1.78,]:
    #for shift in list(np.arange(-3.0, 3.0, 0.01)):
        print('shifting test_dsm_up by: {}'.format(shift))
        test_dsm_shifted = test_dsm + shift
        signed_err = test_dsm_shifted - gt_dsm
        plot_error_map(signed_err, '/data2/kz298/mvs3dm_result/s2p_50_pairs_median.err.jpg',
                       force_range=(-1.5, 1.5))

        # print('median err: {}'.format(np.nanmedian(signed_err)))
        abs_err = np.abs(signed_err)
        median_abs_err = np.nanmedian(abs_err)
        print('median abs_err: {}'.format(median_abs_err))
        completeness = np.sum(abs_err < 1.0) / np.sum(valid_mask)
        print('completeness: {}'.format(completeness))

        if median_abs_err < best_median_abs_err:
            best_shift = shift
            best_median_abs_err = median_abs_err
            best_completeness = completeness
        print('\n')

    print('\n\n')
    print('best shift: {}'.format(best_shift))
    print('best_median_abs_err: {}'.format(best_median_abs_err))
    print('best completeness: {}'.format(best_completeness))
