import numpy as np
import logging
from lib.esti_similarity import esti_similarity
from lib.esti_linear import esti_linear


def check_align(source, target):
    logging.info('\nsource --> target: none')
    err = np.sqrt(np.sum((source - target) ** 2, axis=1))
    logging.info('min: {}, max: {}, mean: {}, median: {}'.format(np.min(err), np.max(err), np.mean(err), np.median(err)))

    logging.info('\nsource --> target: linear')
    M, t = esti_linear(source, target)

    logging.info('\nsource --> target: similarity')
    M, t = esti_similarity(source, target)

    logging.info('\nsource --> target: translation')
    t = np.mean(target - source, axis=0).reshape((1, 3))
    logging.info('translation: {}'.format(t))
    source_aligned = source + np.tile(t, (source.shape[0], 1))
    err = np.sqrt(np.sum((source_aligned - target) ** 2, axis=1))
    logging.info('min: {}, max: {}, mean: {}, median: {}'.format(np.min(err), np.max(err), np.mean(err), np.median(err)))


if __name__ == '__main__':
    pass
