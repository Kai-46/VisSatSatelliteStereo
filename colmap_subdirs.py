import os


def make_subdirs(colmap_dir):
    subdirs = [
                colmap_dir,
                os.path.join(colmap_dir, 'sfm_perspective'),
                os.path.join(colmap_dir, 'sfm_perspective/images'),
                os.path.join(colmap_dir, 'sfm_perspective/init'),
                os.path.join(colmap_dir, 'sfm_perspective/sparse'),
                os.path.join(colmap_dir, 'sfm_perspective/sparse_ba'),
                os.path.join(colmap_dir, 'sfm_pinhole'),
                os.path.join(colmap_dir, 'sfm_pinhole/images'),
                os.path.join(colmap_dir, 'sfm_pinhole/init'),
                os.path.join(colmap_dir, 'sfm_pinhole/sparse'),
                os.path.join(colmap_dir, 'sfm_pinhole/sparse_ba'),
                os.path.join(colmap_dir, 'mvs')
    ]

    for item in subdirs:
        if not os.path.exists(item):
            os.mkdir(item)