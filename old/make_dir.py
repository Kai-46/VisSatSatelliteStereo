import os
import shutil


def make_dir(path, delete_exist=False):
    exist = os.path.exists(path)
    if delete_exist and exist:
        shutil.rmtree(path, ignore_errors=True)
        exist = False

    if not exist:
        os.mkdir(path)
