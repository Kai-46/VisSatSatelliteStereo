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

import logging
import sys


class GlobalLogger(object):
    def __init__(self):
        logging.root.setLevel(logging.INFO)

        self.file_handler = None
        self.stream_handler = None

    def set_log_file(self, log_file):
        if self.file_handler is not None:
            # remove the current logging handler
            logging.root.removeHandler(self.file_handler)

        self.file_handler = logging.FileHandler(log_file, 'w')
        self.file_handler.setLevel(logging.INFO)
        logging.root.addHandler(self.file_handler)

    def turn_off_file_log(self):
        if self.file_handler is not None:
            # remove the current logging handler
            logging.root.removeHandler(self.file_handler)
            self.file_handler = None

    def turn_on_terminal(self):
        self.stream_handler = logging.StreamHandler(sys.stdout)
        self.stream_handler.setLevel(logging.INFO)
        logging.root.addHandler(self.stream_handler)

    def turn_off_terminal(self):
        if self.stream_handler is not None:
            logging.root.removeHandler(self.stream_handler)
            self.stream_handler = None

    def write(self, msg):
        logging.info(msg)
