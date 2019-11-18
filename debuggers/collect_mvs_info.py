#  ===============================================================================================================
#  Copyright (c) 2019, Cornell University. All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without modification, are permitted provided that
#  the following conditions are met:
#
#      * Redistributions of source code must retain the above copyright otice, this list of conditions and
#        the following disclaimer.
#
#      * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and
#        the following disclaimer in the documentation and/or other materials provided with the distribution.
#
#      * Neither the name of Cornell University nor the names of its contributors may be used to endorse or
#        promote products derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
#  WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
#  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE
#  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
#  TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
#  HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
#   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
#  OF SUCH DAMAGE.
#
#  Author: Kai Zhang (kz298@cornell.edu)
#
#  The research is based upon work supported by the Office of the Director of National Intelligence (ODNI),
#  Intelligence Advanced Research Projects Activity (IARPA), via DOI/IBC Contract Number D17PC00287.
#  The U.S. Government is authorized to reproduce and distribute copies of this work for Governmental purposes.
#  ===============================================================================================================


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

