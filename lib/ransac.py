import numpy as np
from lib.procrustes import procrustes
import logging

def esti_simiarity(source, target, samples_per_trial=3, num_of_trials=5000, thres=1.):
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
    source_modified = np.dot(source, transforms[idx]['scale'] * transforms[idx]['rotation']) \
                      + np.tile(transforms[idx]['translation'], (samples_cnt, 1))
    err = np.sqrt(np.sum((source_modified - target) ** 2, axis=1))
    logging.info('\talignment error after, min: {}, max: {}, mean: {}, median: {}'.format(np.min(err), np.max(err), np.mean(err), np.median(err)))


    return transforms[idx]['scale'], transforms[idx]['rotation'], transforms[idx]['translation']


