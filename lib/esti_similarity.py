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
from lib.procrustes import procrustes
import logging


def esti_similarity(source, target):
    assert (source.shape[0] == target.shape[0])

    samples_cnt = source.shape[0]

    _, _, trans = procrustes(target, source, reflection=False)

    scale = trans['scale']
    rotation = trans['rotation']
    t = trans['translation']

    logging.info('\nestimated: ')
    logging.info('scale: {}'.format(scale))
    logging.info('rotation: {}'.format(rotation))
    logging.info('translation: {}'.format(t))

    # make sure we get a rotation matrix
    u, s, vh = np.linalg.svd(rotation)
    cond = s[0] / s[2]
    logging.info('condition number of rotation: {}, smallest singular value: {}'.format(cond, s[2]))

    M = scale * rotation

    # compute re-projection error over the whole set
    err = np.sqrt(np.sum((source - target) ** 2, axis=1))
    logging.info('alignment error before, min: {}, max: {}, mean: {}, median: {}'.format(np.min(err), np.max(err), np.mean(err), np.median(err)))
    source_aligned = np.dot(source, M) + np.tile(t, (samples_cnt, 1))
    err = np.sqrt(np.sum((source_aligned - target) ** 2, axis=1))
    logging.info('alignment error after, min: {}, max: {}, mean: {}, median: {}'.format(np.min(err), np.max(err), np.mean(err), np.median(err)))

    return M, t


def esti_similarity_ransac(source, target, samples_per_trial=3, num_of_trials=5000, thres=1.):
    assert (source.shape[0] == target.shape[0])

    samples_cnt = source.shape[0]

    support_sizes = np.zeros((num_of_trials, ))
    transforms = []

    cnt = 0
    while cnt < num_of_trials:
        subset_idx = []
        while len(subset_idx) < samples_per_trial:
            candi_idx = np.random.randint(samples_cnt)
            if candi_idx not in subset_idx:
                subset_idx.append(candi_idx)

        source_subset = np.array([source[idx, :] for idx in subset_idx])
        target_subset = np.array([target[idx, :] for idx in subset_idx])

        _, source_subset_aligned, trans = procrustes(target_subset, source_subset, reflection=False)

        check_err = np.mean(np.sqrt(np.sum((source_subset_aligned - target_subset) ** 2, axis=1)))

        transforms.append(trans)
        # apply transform to the whole set
        source_aligned = np.dot(source, trans['scale'] * trans['rotation']) \
                          + np.tile(trans['translation'], (samples_cnt, 1))
        # compute residual error
        err = np.sqrt(np.sum((source_aligned - target) ** 2, axis=1))

        support_sizes[cnt] = np.sum(err < thres) / samples_cnt * 100


        logging.info('ransac trial: {} / {}, check_err: {}, thres: {}, support size: {} %'.format(cnt + 1, num_of_trials, check_err, thres, support_sizes[cnt]))

        cnt += 1
    # return the best result
    idx = np.argmax(support_sizes)

    logging.info('ransac summary: ')
    logging.info('\tsamples_per_trial: {}, num_of_trials: {}, thres {}'.format(samples_per_trial, num_of_trials, thres))
    logging.info('\tsupport size, min: {} %, max: {} %'.format(np.min(support_sizes), np.max(support_sizes)))
    # compute re-projection error over the whole set
    err = np.sqrt(np.sum((source - target) ** 2, axis=1))
    logging.info('\talignment error before, min: {}, max: {}, mean: {}, median: {}'.format(np.min(err), np.max(err), np.mean(err), np.median(err)))
    source_aligned = np.dot(source, transforms[idx]['scale'] * transforms[idx]['rotation']) \
                      + np.tile(transforms[idx]['translation'], (samples_cnt, 1))
    err = np.sqrt(np.sum((source_aligned - target) ** 2, axis=1))
    logging.info('\talignment error after, min: {}, max: {}, mean: {}, median: {}'.format(np.min(err), np.max(err), np.mean(err), np.median(err)))


    return transforms[idx]['scale'], transforms[idx]['rotation'], transforms[idx]['translation']


# test the correctness of this algorithm
def rvs(dim=3):
     random_state = np.random
     H = np.eye(dim)
     D = np.ones((dim,))
     for n in range(1, dim):
         x = random_state.normal(size=(dim-n+1,))
         D[n-1] = np.sign(x[0])
         x[0] -= D[n-1]*np.sqrt((x*x).sum())
         # Householder transformation
         Hx = (np.eye(dim-n+1) - 2.*np.outer(x, x)/(x*x).sum())
         mat = np.eye(dim)
         mat[n-1:, n-1:] = Hx
         H = np.dot(H, mat)
         # Fix the last sign such that the determinant is 1
     D[-1] = (-1)**(1-(dim % 2))*D.prod()
     # Equivalent to np.dot(np.diag(D), H) but faster, apparently
     H = (D*H.T).T
     return H


if __name__ == '__main__':
    from lib.logger import GlobalLogger
    logger = GlobalLogger()
    logger.turn_on_terminal()

    R = rvs(3)
    c = (10 * np.random.randn())**2
    t = 5 * np.random.randn(1, 3)

    print('ground-truth: ')
    print('scale: {}'.format(c))
    print('rotation: \n{}'.format(R))
    print('translation: {}'.format(t))

    cnt = 500
    input = np.random.randn(cnt, 3)
    target = np.dot(input, c*R) + np.tile(t, (cnt, 1))
    M, t_hat = esti_similarity(input, target)

    # check the reproj error
    err = np.sqrt(np.sum((input - target) ** 2, axis=1))
    print('\nbefore alignment, mean, median err: {}, {}'.format(np.mean(err), np.median(err)))

    input_aligned = np.dot(input, M) + np.tile(t_hat, (cnt, 1))
    err = np.sqrt(np.sum((input_aligned - target) ** 2, axis=1))
    print('after alignment, mean, median err: {}, {}'.format(np.mean(err), np.median(err)))