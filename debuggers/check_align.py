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
