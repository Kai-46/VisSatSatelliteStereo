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

import os
import json
import colmap.database as database
from colmap.extract_sfm import extract_camera_dict


def convert_colmap_sfm_to_template(sfm_dir, camera_model, template_file):
    assert(camera_model == 'PINHOLE' or camera_model == 'PERSPECTIVE')

    camera_dict = extract_camera_dict(sfm_dir)

    if camera_model == 'PINHOLE':
        write_template_pinhole(camera_dict, template_file)
    else:
        write_template_perspective(camera_dict, template_file)


def write_template_perspective(perspective_dict, template_file):
    template = {}
    cameras_line_template = '{camera_id} PERSPECTIVE {width} {height} {fx} {fy} {cx} {cy} {s}\n'
    images_line_template = '{image_id} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {camera_id} {image_name}\n\n'

    for img_name in perspective_dict:
        # w, h, fx, fy, cx, cy, s, qvec, t
        params = perspective_dict[img_name]
        w = params[0]
        h = params[1]
        fx = params[2]
        fy = params[3]
        cx = params[4]
        cy = params[5]
        s = params[6]
        qvec = params[7:11]
        tvec = params[11:14]

        # write_to_template
        cam_line = cameras_line_template.format(camera_id="{camera_id}", width=w, height=h,
                                               fx=fx, fy=fy, cx=cx, cy=cy, s=s)
        img_line = images_line_template.format(image_id="{image_id}", qw=qvec[0], qx=qvec[1], qy=qvec[2], qz=qvec[3],
                                            tx=tvec[0], ty=tvec[1], tz=tvec[2], camera_id="{camera_id}", image_name=img_name)
        template[img_name] = (cam_line, img_line)

    with open(template_file, 'w') as fp:
         json.dump(template, fp, indent=2)


def write_template_pinhole(pinhole_dict, template_file):
    template = {}
    cameras_line_template = '{camera_id} PINHOLE {width} {height} {fx} {fy} {cx} {cy}\n'
    images_line_template = '{image_id} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {camera_id} {image_name}\n\n'

    for img_name in pinhole_dict:
        # w, h, fx, fy, cx, cy, qvec, t
        params = pinhole_dict[img_name]
        w = params[0]
        h = params[1]
        fx = params[2]
        fy = params[3]
        cx = params[4]
        cy = params[5]
        qvec = params[6:10]
        tvec = params[10:13]

        # write_to_template
        # construct a pinhole camera
        cam_line = cameras_line_template.format(camera_id="{camera_id}", width=w, height=h, fx=fx, fy=fy, cx=cx, cy=cy)
        img_line = images_line_template.format(image_id="{image_id}", qw=qvec[0], qx=qvec[1], qy=qvec[2], qz=qvec[3],
                                            tx=tvec[0], ty=tvec[1], tz=tvec[2], camera_id="{camera_id}", image_name=img_name)
        template[img_name] = (cam_line, img_line)

    with open(template_file, 'w') as fp:
         json.dump(template, fp, indent=2)


def create_init_files(db_file, template_file, out_dir):
    # read database
    db = database.COLMAPDatabase.connect(db_file)
    table_images = db.execute("SELECT * FROM images")
    img_name2id_dict = {}
    for row in table_images:
        img_name2id_dict[row[1]] = row[0]

    # load template
    with open(template_file) as fp:
        template = json.load(fp)

    cameras_txt_lines = []
    images_txt_lines = []
    for img_name, img_id in img_name2id_dict.items():
        camera_line = template[img_name][0].format(camera_id=img_id)
        cameras_txt_lines.append(camera_line)

        image_line = template[img_name][1].format(image_id=img_id, camera_id=img_id)
        images_txt_lines.append(image_line)

    with open(os.path.join(out_dir, 'cameras.txt'), 'w') as fp:
        fp.writelines(cameras_txt_lines)

    with open(os.path.join(out_dir, 'images.txt'), 'w') as fp:
        fp.writelines(images_txt_lines)
        fp.write('\n')

    # create an empty points3D.txt
    fp = open(os.path.join(out_dir, 'points3D.txt'), 'w')
    fp.close()

    # add inspector
    with open(os.path.join(out_dir, 'img_name2id.txt'), 'w') as fp:
        fp.write('# template_file: {}\n'.format(os.path.abspath(template_file)))
        fp.write('# db_file: {}\n'.format(os.path.abspath(db_file)))
        fp.write('# format: img_name colmap_id\n')
        for img_name in sorted(img_name2id_dict.keys()):
            fp.write('{} {}\n'.format(img_name, img_name2id_dict[img_name]))
