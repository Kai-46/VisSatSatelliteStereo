import os
import json
import cv2
import numpy as np
import imageio
import colmap.database as database


def make_subdirs(out_dir):
    subdirs = [ os.path.join(out_dir, 'images'),
                os.path.join(out_dir, 'init'),
                os.path.join(out_dir, 'sparse'),
                os.path.join(out_dir, 'sparse_ba'),
                os.path.join(out_dir, 'dense') ]

    for item in subdirs:
        if not os.path.exists(item):
            os.mkdir(item)

    return subdirs


def prep_for_colmap(tile_dir, out_dir):
    subdirs = make_subdirs(out_dir)

    image_subdir = subdirs[0]
    init_subdir = subdirs[1]

    # open perspective approximation file
    with open(os.path.join(tile_dir, 'approx_perspective_utm.json')) as fp:
        perspective_dict = json.load(fp)

    template = {}

    # skew-correct the images
    cameras_line_template = '{camera_id} PINHOLE {width} {height} {fx} {fy} {cx} {cy}\n'
    images_line_template = '{image_id} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {camera_id} {image_name}\n\n'

    for item in os.listdir(os.path.join(tile_dir, 'images')):
        # fx, fy, cx, cy, s, qvec, t
        params = perspective_dict[item]
        fx = params[0]
        fy = params[1]
        cx = params[2]
        cy = params[3]
        s = params[4]
        qvec = params[5:9]
        tvec = params[9:12]

        # compute homography and update s, cx
        norm_skew = s / fy
        s = 0.
        cx = cx - s * cy / fy

        print('removing skew, image: {}, normalized skew: {}'.format(item, norm_skew))
        homography = np.array([[1, -norm_skew, 0],
                               [0, 1, 0],
                               [0, 0, 1]])

        # load source image
        im_src = imageio.imread(os.path.join(tile_dir, 'images', item)).astype(dtype=np.float64)
        height, width = im_src.shape

        # compute bounding box after applying the above homography
        points = np.dot(homography, np.array([[0., width, width, 0.],
                                              [0., 0., height, height],
                                              [1., 1., 1., 1.]]))
        w = int(np.min((points[0, 1], points[0, 2])))
        h = int(np.min((points[1, 2], points[1, 3])))
        print('original image size, width: {}, height: {}'.format(width, height))
        print('skew-corrected image size, width: {}, height: {}'.format(w, h))

        # warp image
        img_dst = cv2.warpPerspective(im_src, homography, (w, h))
        imageio.imwrite(os.path.join(image_subdir, 'images', item), img_dst.astype(dtype=np.uint8))

        # write to template
        cam_line = cameras_line_template.format(camera_id="{camera_id}", width=w, height=h,
                                            fx=fx, fy=fy, cx=cx, cy=cy)
        img_line = images_line_template.format(image_id="{image_id}", qw=qvec[0], qx=qvec[1], qy=qvec[2], qz=qvec[3],
                                            tx=tvec[0], ty=tvec[1], tz=tvec[2], camera_id="{camera_id}",
                                            image_name=item)
        template[item] = (cam_line, img_line)

    with open(init_subdir, 'template.json') as fp:
        json.dump(template, fp)


def create_init_files(colmap_dir):
    # read database
    db_path = os.path.join(colmap_dir, 'database.db')
    db = database.COLMAPDatabase.connect(db_path)
    table_images = db.execute("SELECT * FROM images")
    img_name2id_dict = {}
    for row in table_images:
        img_name2id_dict[row[1]] = row[0]

    init_dir = os.path.join(colmap_dir, 'init')
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