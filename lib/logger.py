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
