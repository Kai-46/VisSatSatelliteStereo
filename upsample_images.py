import cv2
import json
import os
import copy


def upsample_images(skew_correct_dir, out_dir):
    with open(os.path.join(skew_correct_dir, 'pinhole_dict.json')) as fp:
        orig_pinhole_dict = json.load(fp)

    for dir in [out_dir, os.path.join(out_dir, 'images')]:
        if not os.path.exists(dir):
            os.mkdir(dir)

    new_pinhole_dict = copy.deepcopy(orig_pinhole_dict)
    for img_name in sorted(orig_pinhole_dict.keys()):
        params = orig_pinhole_dict[img_name]

        orig_width = params[0]
        orig_height = params[1]

        orig_img = cv2.imread(os.path.join(skew_correct_dir, 'images', img_name))

        # resize image
        scale = 2
        new_width = scale * orig_width
        new_height = scale * orig_height
        new_img = cv2.resize(orig_img, (new_width, new_height), interpolation=cv2.INTER_AREA)

        # modify params
        params[0] = new_width
        params[1] = new_height
        # change fx, fy, cx, cy
        params[2] = params[2] * scale
        params[3] = params[3] * scale
        params[4] = params[4] * scale
        params[5] = params[5] * scale

        # save image
        cv2.imwrite(os.path.join(out_dir, 'images', img_name), new_img)
        new_pinhole_dict[img_name] = params

    with open(os.path.join(out_dir, 'pinhole_dict.json'), 'w') as fp:
        json.dump(new_pinhole_dict, fp, indent=2, sort_keys=True)


if __name__ == '__main__':
    work_dir = '/data2/kz298/mvs3dm_result/MasterProvisional2'
    skew_correct_dir = os.path.join(work_dir, 'colmap/skew_correct')
    out_dir = os.path.join(work_dir, 'colmap/upsample')
    upsample_images(skew_correct_dir, out_dir)
