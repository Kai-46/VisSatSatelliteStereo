import lib.database
import sys
import os
import json

if __name__ == '__main__':
    proj_dir = sys.argv[1]
    out_dir = os.path.join(proj_dir, 'init')
    print('proj_dir: {}, out_dir: {}'.format(proj_dir, out_dir))
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    # read database
    db = lib.database.COLMAPDatabase.connect(os.path.join(proj_dir, 'database.db'))
    table_images = db.execute("SELECT * FROM images")
    img_name2id_dict = {}
    for row in table_images:
        img_name2id_dict[row[1]] = row[0]

    # load camera_data.json
    with open(os.path.join(proj_dir, 'camera_data.json')) as fp:
        camera_dict = json.load(fp)

    cameras_txt_lines = []
    images_txt_lines = []
    for img_name, img_id in img_name2id_dict.items():
        camera_line = camera_dict[img_name][0].format(camera_id=img_id)
        cameras_txt_lines.append(camera_line)

        image_line = camera_dict[img_name][1].format(image_id=img_id, camera_id=img_id)
        images_txt_lines.append(image_line)

    with open(os.path.join(out_dir, 'cameras.txt'), 'w') as fp:
        fp.writelines(cameras_txt_lines)

    with open(os.path.join(out_dir, 'images.txt'), 'w') as fp:
        fp.writelines(images_txt_lines)
        fp.write('\n')

    # create an empty points3D.txt
    fp = open(os.path.join(out_dir, 'points3D.txt'), 'w')
    fp.close()

    # create sparse directory
    sparse_dir = os.path.join(proj_dir, 'sparse')
    if not os.path.exists(sparse_dir):
        os.mkdir(sparse_dir)

    sparse_ba_dir = os.path.join(proj_dir, 'sparse_ba')
    if not os.path.exists(sparse_ba_dir):
        os.mkdir(sparse_ba_dir)

    dense_dir = os.path.join(proj_dir, 'dense')
    if not os.path.exists(dense_dir):
        os.mkdir(dense_dir)