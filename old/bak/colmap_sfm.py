import os
# import colmap_sfm_perspective
# import colmap_sfm_pinhole


def make_subdirs(colmap_dir):
    subdirs = [
                colmap_dir,
                os.path.join(colmap_dir, 'sfm_perspective'),
                os.path.join(colmap_dir, 'sfm_pinhole'),
                os.path.join(colmap_dir, 'mvs')
    ]

    for item in subdirs:
        if not os.path.exists(item):
            os.mkdir(item)


# def run(work_dir):
#     colmap_dir = os.path.join(work_dir, 'colmap')
#     if not os.path.exists(colmap_dir):
#         os.mkdir(colmap_dir)
#
#     init_camera_file =