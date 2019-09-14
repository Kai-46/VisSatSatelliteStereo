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


def collect_mvs_info(work_dir):
    mvs_dir = os.path.join(work_dir, 'colmap/mvs')
    out_dir = os.path.join(work_dir, 'mvs_results')

    # read projection matrix and save to json
    proj_mats = {}
    with open(os.path.join(mvs_dir, 'proj_mats.txt')) as fp:
        for line in fp.readlines():
            items = line.strip().split(' ')
            img_name = items[0]
            mat = [float(items[i]) for i in range(1, len(items))]
            proj_mats[img_name] = mat

    with open(os.path.join(out_dir, 'proj_mats.json'), 'w') as fp:
        json.dump(proj_mats, fp, indent=2, sort_keys=True)

    # read inverse_projection matrix and save to json
    inv_proj_mats = {}
    with open(os.path.join(mvs_dir, 'inv_proj_mats.txt')) as fp:
        for line in fp.readlines():
            items = line.strip().split(' ')
            img_name = items[0]
            mat = [float(items[i]) for i in range(1, len(items))]
            inv_proj_mats[img_name] = mat

    with open(os.path.join(out_dir, 'inv_proj_mats.json'), 'w') as fp:
        json.dump(inv_proj_mats, fp, indent=2, sort_keys=True)

    # read img_idx2name, read ref2src, and save to json
    img_idx2name = {}
    with open(os.path.join(mvs_dir, 'img_idx2name.txt')) as fp:
        for line in fp.readlines():
            items = line.strip().split(' ')
            img_idx = int(items[0])
            img_name = items[1]
            img_idx2name[img_idx] = img_name

    ref2src = {}
    with open(os.path.join(mvs_dir, 'ref2src.txt')) as fp:
        lines = fp.readlines()
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line:
                idx = line.find(':')
                ref_img_id = int(line[idx+1:])
                ref_img_name = img_idx2name[ref_img_id]

                line = lines[i + 2].strip()     # src images
                idx = line.find(':')
                src_img_idxs = line[idx+1:].strip().split(' ')
                src_img_idxs = [int(idx) for idx in src_img_idxs]
                src_img_names = [img_idx2name[idx] for idx in src_img_idxs]

                ref2src[ref_img_name] = {'num_of_src': len(src_img_idxs),
                                         'src_img_names': src_img_names}

                i += 3
            else:
                i += 1

    with open(os.path.join(out_dir, 'ref2src.json'), 'w') as fp:
        json.dump(ref2src, fp, indent=2, sort_keys=True)


if __name__ == '__main__':
    work_dir = '/data2/kz298/core3d_result/aoi-d4-jacksonville'
    collect_mvs_info(work_dir)

