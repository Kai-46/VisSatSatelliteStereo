import os
import json
import numpy as np
import imageio
import colmap.database as database
import logging
import shutil
from colmap.read_model import read_model
from lib.warp_affine import warp_affine
import colmap_subdirs


def prep_for_sfm_perspective(tile_dir, colmap_dir):
    colmap_subdirs.make_subdirs(colmap_dir)

    sfm_dir = os.path.join(colmap_dir, 'sfm_perspective')

    # delete the database, otherwise colmap will skip the feature extraction step
    db_path = os.path.join(sfm_dir, 'database.db')
    if os.path.exists(db_path):
        os.remove(db_path)

    # copy images
    image_subdir = os.path.join(sfm_dir, 'images')
    if os.path.exists(image_subdir):
        shutil.rmtree(image_subdir, ignore_errors=True)
    shutil.copytree(os.path.join(tile_dir, 'images'), image_subdir)

    # write init files
    init_subdir = os.path.join(sfm_dir, 'init')
    with open(os.path.join(tile_dir, 'approx_perspective_utm.json')) as fp:
        perspective_dict = json.load(fp)

    template = {}

    cameras_line_template = '{camera_id} PERSPECTIVE {width} {height} {fx} {fy} {cx} {cy} {s}\n'
    images_line_template = '{image_id} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {camera_id} {image_name}\n\n'

    for img_name in os.listdir(image_subdir):
        # fx, fy, cx, cy, s, qvec, t
        params = perspective_dict[img_name]
        fx = params[0]
        fy = params[1]
        cx = params[2]
        cy = params[3]
        s = params[4]
        qvec = params[5:9]
        tvec = params[9:12]

        img = imageio.imread(os.path.join(image_subdir, img_name))
        h, w = img.shape

        # write_to_template
        cam_line = cameras_line_template.format(camera_id="{camera_id}", width=w, height=h,
                                               fx=fx, fy=fy, cx=cx, cy=cy, s=s)
        img_line = images_line_template.format(image_id="{image_id}", qw=qvec[0], qx=qvec[1], qy=qvec[2], qz=qvec[3],
                                            tx=tvec[0], ty=tvec[1], tz=tvec[2], camera_id="{camera_id}", image_name=img_name)
        template[img_name] = (cam_line, img_line)

    with open(os.path.join(init_subdir, 'template.json'), 'w') as fp:
         json.dump(template, fp, indent=2)


def prep_for_sfm_pinhole(colmap_dir):
    colmap_subdirs.make_subdirs(colmap_dir)

    # delete the database, otherwise colmap will skip the feature extraction step
    # db_path = os.path.join(colmap_dir, 'sfm_pinhole/database.db')
    # if os.path.exists(db_path):
    #     os.remove(db_path)

    # read sfm_perspective
    colmap_cameras, colmap_images, _ = read_model(os.path.join(colmap_dir, 'sfm_perspective/sparse_ba'), '.txt')

    # warp image and write init template
    image_subdir = os.path.join(colmap_dir, 'sfm_pinhole/images')
    init_subdir = os.path.join(colmap_dir, 'sfm_pinhole/init')

    if os.path.exists(image_subdir):
        shutil.rmtree(image_subdir, ignore_errors=True)
    os.mkdir(image_subdir)

    template = {}
    cameras_line_template = '{camera_id} PINHOLE {width} {height} {fx} {fy} {cx} {cy}\n'
    images_line_template = '{image_id} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {camera_id} {image_name}\n\n'

    affine_warpings = {}

    for img_id in colmap_images:
        image = colmap_images[img_id]
        img_name = image.name
        qvec = image.qvec
        tvec = image.tvec
        cam_id = image.camera_id
        cam = colmap_cameras[cam_id]
        params = cam.params
        # fx, fy, cx, cy, s
        fx = params[0]
        fy = params[1]
        cx = params[2]
        cy = params[3]
        s = params[4]

        # compute homography and update s, cx
        norm_skew = s / fy
        cx = cx - s * cy / fy
        # s = 0.

        # warp image
        affine_matrix = np.array([[1, -norm_skew, 0],
                                  [0, 1, 0]])
        img_src = imageio.imread(os.path.join(colmap_dir, 'sfm_perspective/images/{}'.format(img_name)))
        img_dst, off_set, affine_matrix = warp_affine(img_src, affine_matrix)
        imageio.imwrite(os.path.join(image_subdir, '{}'.format(img_name)), img_dst)

        affine_warpings[img_name] = affine_matrix

        new_height, new_width = img_dst.shape

        logging.info('removed normalized skew: {} in image: {}, new image size: {}, {}'.format(norm_skew, img_name, new_width, new_height))

        # add off_set to camera parameters
        cx += off_set[0]
        cy += off_set[1]

        # construct a pinhole camera
        cam_line = cameras_line_template.format(camera_id="{camera_id}", width=new_width, height=new_height, fx=fx, fy=fy, cx=cx, cy=cy)
        img_line = images_line_template.format(image_id="{image_id}", qw=qvec[0], qx=qvec[1], qy=qvec[2], qz=qvec[3],
                                            tx=tvec[0], ty=tvec[1], tz=tvec[2], camera_id="{camera_id}", image_name=img_name)
        template[img_name] = (cam_line, img_line)

    with open(os.path.join(init_subdir, 'template.json'), 'w') as fp:
         json.dump(template, fp, indent=2)

    with open(os.path.join(colmap_dir, 'sfm_pinhole/affine_warpings.txt'), 'w') as fp:
        for img_name in sorted(affine_warpings.keys()):
            matrix = affine_warpings[img_name]
            fp.write('{} {} {} {} {} {} {}\n'.format(img_name, matrix[0, 0], matrix[0, 1], matrix[0, 2],
                                                     matrix[1, 0], matrix[1, 1], matrix[1,2]))


def create_init_files(sfm_dir):
    # read database
    db_path = os.path.join(sfm_dir, 'database.db')
    db = database.COLMAPDatabase.connect(db_path)
    table_images = db.execute("SELECT * FROM images")
    img_name2id_dict = {}
    for row in table_images:
        img_name2id_dict[row[1]] = row[0]

    init_dir = os.path.join(sfm_dir, 'init')
    # load template
    with open(os.path.join(init_dir, 'template.json')) as fp:
        template = json.load(fp)

    cameras_txt_lines = []
    images_txt_lines = []
    for img_name, img_id in img_name2id_dict.items():
        camera_line = template[img_name][0].format(camera_id=img_id)
        cameras_txt_lines.append(camera_line)

        image_line = template[img_name][1].format(image_id=img_id, camera_id=img_id)
        images_txt_lines.append(image_line)

    with open(os.path.join(init_dir, 'cameras.txt'), 'w') as fp:
        fp.writelines(cameras_txt_lines)

    with open(os.path.join(init_dir, 'images.txt'), 'w') as fp:
        fp.writelines(images_txt_lines)
        fp.write('\n')

    # create an empty points3D.txt
    fp = open(os.path.join(init_dir, 'points3D.txt'), 'w')
    fp.close()


if __name__ == '__main__':
    work_dirs = ['/data2/kz298/mvs3dm_result/Explorer',
                '/data2/kz298/mvs3dm_result/MasterProvisional1',
                '/data2/kz298/mvs3dm_result/MasterProvisional2',
                '/data2/kz298/mvs3dm_result/MasterProvisional3',
                '/data2/kz298/mvs3dm_result/MasterSequestered1',
                '/data2/kz298/mvs3dm_result/MasterSequestered2',
                '/data2/kz298/mvs3dm_result/MasterSequestered3',
                '/data2/kz298/mvs3dm_result/MasterSequesteredPark']
    colmap_dirs = [os.path.join(work_dir, 'colmap') for work_dir in work_dirs]

    for colmap_dir in colmap_dirs:
        prep_for_sfm_pinhole(colmap_dir)
