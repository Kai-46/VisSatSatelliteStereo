import tarfile
import os


def create_tar(source_dir, tar_file):
    with tarfile.open(os.path.join(tar_file), 'w') as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))
