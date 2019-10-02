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
import tarfile
import shutil
import unicodedata
import logging

# first find .NTF file, and extract order_id, prod_id, standard name
# then extract rpc file and preview image from the .tar file

# def raw_name_to_cleaned_name(raw_name):
#     idx = raw_name.find('-P1BS-')
#     img_name = raw_name[idx - 13:idx + 26]
#
#     idx = raw_name.find('WV')
#     sensor = raw_name[idx:idx + 4]
#
#     # prepend sensor to img_name
#     cleaned_name = sensor + '_' + img_name
#
#     return cleaned_name


def clean_image_info(file_name):
    file_name = os.path.basename(file_name)
    # get order_id, prod_id
    idx = file_name.find('-P1BS-')
    order_id = file_name[idx + 6:idx + 21]
    prod_id = file_name[idx + 6:idx + 26]
    img_name = file_name[idx - 13:idx + 26]
    return img_name, order_id, prod_id


def process_clean_data_item(item, dataset_dir, out_dir, tmp_dir):
    if item[-4:] == '.NTF' and os.path.exists(os.path.join(dataset_dir, '{}.tar'.format(item[:-4]))):
        logging.info('cleaning {}'.format(item))
        img_name, order_id, prod_id = clean_image_info(item)
        os.symlink(os.path.join(dataset_dir, item), os.path.join(out_dir, '{}.NTF'.format(img_name)))
        tar = tarfile.open(os.path.join(dataset_dir, '{}.tar'.format(item[:-4])))
        tar.extractall(os.path.join(tmp_dir, img_name))

        subfolder = 'DVD_VOL_1'
        for x in os.listdir(os.path.join(tmp_dir, img_name, order_id)):
            if 'DVD_VOL' in x:
                subfolder = x
                break

        des_folder = os.path.join(tmp_dir, img_name, order_id, subfolder, order_id)
        # walk through des_folder
        # img_files = []
        # for root, dirs, files in os.walk(des_folder):
        #     img_files.extend([os.path.join(root, x) for x in files
        #                       if img_name in x and (x[-4:] == '.XML' or x[-4:] == '.JPG')])

        rpc_file = os.path.join(des_folder, '{}_PAN'.format(prod_id), '{}.XML'.format(img_name))
        jpg_file = os.path.join(des_folder, '{}_PAN'.format(prod_id), '{}-BROWSE.JPG'.format(img_name))
        img_files = [rpc_file, jpg_file]
        for x in img_files:
            shutil.copy(x, out_dir)

        # remove control characters in the xml file
        rpc_file = os.path.join(out_dir, '{}.XML'.format(img_name))
        with open(rpc_file, encoding='utf-8', errors='ignore') as fp:
            content = fp.read()
        content = "".join([ch for ch in content if unicodedata.category(ch)[0] != "C"])
        with open(rpc_file, 'w') as fp:
            fp.write(content)
        return True
    return False


def clean_data(dataset_dirs, out_dir, pairing=None):
    # out_dir must exist and be empty
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    dataset_dirs = [os.path.abspath(dataset_dir) for dataset_dir in dataset_dirs]
    logging.info('dataset path: {}'.format(dataset_dirs))
    logging.info('will save files to folder: {}'.format(out_dir))
    logging.info('the standard format is: <7 char date><6 char time>-P1BS-<20 char product id>.NTF\n\n')

    tmp_dir = os.path.join(out_dir, 'tmp')
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.mkdir(tmp_dir)

    cnt = 0
    if pairing is not None:
        for p in pairing:
            pan_ntf = p[0]
            item = os.path.basename(pan_ntf)
            dataset_dir = os.path.dirname(pan_ntf)
            if process_clean_data_item(item, dataset_dir, out_dir, tmp_dir):
                cnt += 1
    else:
        for dataset_dir in sorted(dataset_dirs):
            for item in sorted(os.listdir(dataset_dir)):
                # if 'WV03' not in item:  # only select 'WV03' satellite images
                #     continue
                if process_clean_data_item(item, dataset_dir, out_dir, tmp_dir):
                    cnt += 1

    logging.info('processed {} items in total'.format(cnt))
    # remove tmp_dir
    shutil.rmtree(tmp_dir)


if __name__ == '__main__':
    def main():
        dataset_dir = '/data2/kz298/core3d_pan/jacksonville'
        out_dir = os.path.join(dataset_dir, 'cleaned_data')
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        clean_data(dataset_dir, out_dir)
    main()
