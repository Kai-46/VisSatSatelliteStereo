import os
import json
# import cv2
import numpy as np
import imageio
import colmap.database as database
import logging
import shutil
from colmap.read_model import read_model
from lib.warp_affine import warp_affine

def make_subdirs(colmap_dir):
    subdirs = [ os.path.join(colmap_dir, 'images'),
                os.path.join(colmap_dir, 'init'),
                os.path.join(colmap_dir, 'sparse'),
                os.path.join(colmap_dir, 'sparse_ba'),
                os.path.join(colmap_dir, 'dense'),
                os.path.join(colmap_dir, 'dense/images'),
                os.path.join(colmap_dir, 'dense/sparse'),
                os.path.join(colmap_dir, 'dense/stereo'),
                os.path.join(colmap_dir, 'dense/stereo/depth_maps'),
                os.path.join(colmap_dir, 'dense/stereo/normal_maps'),
                os.path.join(colmap_dir, 'dense/stereo/consistency_graphs')
    ]

    for item in subdirs:
        if not os.path.exists(item):
            os.mkdir(item)

    return subdirs


def prep_for_sfm(tile_dir, colmap_dir):
    subdirs = make_subdirs(colmap_dir)

    image_subdir = subdirs[0]
    init_subdir = subdirs[1]

    # copy images
    if os.path.exists(image_subdir):
        shutil.rmtree(image_subdir)
    shutil.copytree(os.path.join(tile_dir, 'images'), image_subdir)

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


# def prep_for_sfm_pinhole(tile_dir, out_dir):
#     subdirs = make_subdirs(out_dir)
#
#     image_subdir = subdirs[0]
#     init_subdir = subdirs[1]
#
#     # open perspective approximation file
#     with open(os.path.join(tile_dir, 'approx_perspective_utm.json')) as fp:
#         perspective_dict = json.load(fp)
#
#     template = {}
#
#     # skew-correct the images
#     cameras_line_template = '{camera_id} PINHOLE {width} {height} {fx} {fy} {cx} {cy}\n'
#     images_line_template = '{image_id} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {camera_id} {image_name}\n\n'
#
#     for item in os.listdir(os.path.join(tile_dir, 'images')):
#         # fx, fy, cx, cy, s, qvec, t
#         params = perspective_dict[item]
#         fx = params[0]
#         fy = params[1]
#         cx = params[2]
#         cy = params[3]
#         s = params[4]
#         qvec = params[5:9]
#         tvec = params[9:12]
#
#         # compute homography and update s, cx
#         norm_skew = s / fy
#         cx = cx - s * cy / fy
#         # s = 0.
#
#         logging.info('\nremoving skew, image: {}, normalized skew: {}'.format(item, norm_skew))
#         homography = np.array([[1, -norm_skew, 0],
#                                [0, 1, 0],
#                                [0, 0, 1]])
#
#         # load source image
#         im_src = imageio.imread(os.path.join(tile_dir, 'images', item)).astype(dtype=np.float64)
#         height, width = im_src.shape
#
#         # compute bounding box after applying the above homography
#         points = np.dot(homography, np.array([[0., width, width, 0.],
#                                               [0., 0., height, height],
#                                               [1., 1., 1., 1.]]))
#         w = int(np.min((points[0, 1], points[0, 2])))
#         h = int(np.min((points[1, 2], points[1, 3])))
#         logging.info('original image size, width: {}, height: {}'.format(width, height))
#         logging.info('skew-corrected image size, width: {}, height: {}'.format(w, h))
#
#         # warp image
#         img_dst = cv2.warpPerspective(im_src, homography, (w, h))
#         imageio.imwrite(os.path.join(image_subdir, item), img_dst.astype(dtype=np.uint8))
#
#         # write to template
#         cam_line = cameras_line_template.format(camera_id="{camera_id}", width=w, height=h,
#                                             fx=fx, fy=fy, cx=cx, cy=cy)
#         img_line = images_line_template.format(image_id="{image_id}", qw=qvec[0], qx=qvec[1], qy=qvec[2], qz=qvec[3],
#                                             tx=tvec[0], ty=tvec[1], tz=tvec[2], camera_id="{camera_id}",
#                                             image_name=item)
#         template[item] = (cam_line, img_line)
#
#     with open(os.path.join(init_subdir, 'template.json'), 'w') as fp:
#         json.dump(template, fp, indent=2)


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


def prep_for_mvs(colmap_dir):
    # read sparse reconstruction result
    colmap_cameras, colmap_images, colmap_points3D = read_model(os.path.join(colmap_dir, 'sparse_ba'), '.bin')

    cameras_txt_lines = []
    with open(os.path.join(colmap_dir, 'dense/sparse/images.txt'), 'w') as fp:
        # write comment
        comment = '# Image list with two lines of data per image:\
        #   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\
        #   POINTS2D[] as (X, Y, POINT3D_ID)\n'
        fp.write(comment)

        for img_id in colmap_images:
            image = colmap_images[img_id]
            img_name = image.name
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

            logging.info('removing normalized skew: {} in image: {}'.format(norm_skew, img_name))

            # warp image
            affine_matrix = np.array([[1, -norm_skew, 0],
                                      [0, 1, 0]])
            img_src = imageio.imread(os.path.join(colmap_dir, 'images/{}'.format(img_name)))
            img_dst, off_set, size = warp_affine(img_src, affine_matrix)
            imageio.imwrite(os.path.join(colmap_dir, 'dense/images/{}'.format(img_name)), img_dst)

            # construct a pinhole camera
            line = '{cam_id} PINHOLE {width} {height} {fx} {fy} {cx} {cy}\n'.format(
                cam_id=cam_id, width=size[0], height=size[1], fx=fx, fy=fy, cx=cx, cy=cy
            )
            cameras_txt_lines.append(line)

            first_line = '{img_id} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {cam_id} {img_name}\n'.format(
                img_id=img_id, qw=image.qvec[0], qx=image.qvec[1], qy=image.qvec[2], qz=image.qvec[3],
                tx=image.tvec[0], ty=image.tvec[1], tz=image.tvec[2], cam_id=cam_id, img_name=img_name
            )
            fp.write(first_line)

            second_line = ''
            # modify image key points
            for i in range(image.xys.shape[0]):
                col = image.xys[i, 0]
                row = image.xys[i, 1]

                # apply affine transformation
                tmp = np.dot(affine_matrix, np.array([col, row, 1]).reshape(-1, 1))
                col = tmp[0, 0] + off_set[0]
                row = tmp[1, 0] + off_set[1]

                second_line += ' {col} {row} {point3d_id}'.format(col=col, row=row, point3d_id=image.point3D_ids[i])
            second_line = second_line[1:] + '\n'
            fp.write(second_line)

    with open(os.path.join(colmap_dir, 'dense/sparse/cameras.txt'), 'w') as fp:
        comment = '# Camera list with one line of data per camera:\
        #   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n'
        fp.write(comment)
        fp.writelines(cameras_txt_lines)

    with open(os.path.join(colmap_dir, 'dense/sparse/points3D.txt'), 'w') as fp:
        comment = '# 3D point list with one line of data per point: \
        #   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n'
        fp.write(comment)
        for point3d_id in colmap_points3D:
            point3d = colmap_points3D[point3d_id]
            line = '{point3d_id} {x} {y} {z} {r} {g} {b} {err}'.format(
                point3d_id=point3d_id, x=point3d.xyz[0], y=point3d.xyz[1], z=point3d.xyz[2],
                r=point3d.rgb[0], g=point3d.rgb[1], b=point3d.rgb[2], err=point3d.error
            )

            for i in range(len(point3d.image_ids)):
                line += ' {img_id} {point2d_idx}'.format(img_id=point3d.image_ids[i], point2d_idx=point3d.point2D_idxs[i])
            line += '\n'
            fp.write(line)


if __name__ == '__main__':
    work_dir = '/data2/kz298/core3d_aoi/aoi-d4-jacksonville'
    # prepare colmap workspace
    colmap_dir = os.path.join(work_dir, 'colmap')
    if not os.path.exists(colmap_dir):
        os.mkdir(colmap_dir)
    prep_for_sfm(work_dir, colmap_dir)

    prep_for_mvs(colmap_dir)