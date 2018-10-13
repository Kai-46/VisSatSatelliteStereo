#!/usr/bin/env python

import os
import sys
import tarfile
import shutil

# first find .NTF file, and extract order_id, prod_id, standard name
# then extract rpc file and preview image from the .tar file
# save hard links to the .NTF file in 'cleaned_data/'


def prep_data(path):
    path = os.path.abspath(path)
    print('dataset path: {}'.format(path))
    print('will save files to the subfolder: cleaned_data/')
    print('the standard format is: <7 char date><6 char time>-P1BS-<20 char product id>.NTF\n\n')

    tmp_folder = os.path.join(path, 'tmp')
    if os.path.exists(tmp_folder):
        shutil.rmtree(tmp_folder, ignore_errors=True)
    os.mkdir(tmp_folder)

    out_folder = os.path.join(path, 'cleaned_data')
    if os.path.exists(out_folder):
        shutil.rmtree(out_folder, ignore_errors=True)
    os.mkdir(out_folder)

    for item in sorted(os.listdir(path)):
        if item[-4:] == '.NTF':
            # get order_id, prod_id
            idx = item.find('-P1BS-')
            order_id = item[idx+6:idx+21]
            prod_id = item[idx+6:idx+26]
            img_name = item[idx-13:idx+26]

            # rename NTF
            # os.rename(os.path.join(path, item), os.path.join(path, '{}.NTF'.format(img_name)))

            os.link(os.path.join(path, item), os.path.join(out_folder, '{}.NTF'.format(img_name)))

            tar = tarfile.open(os.path.join(path, '{}.tar'.format(item[:-4])))

            tar.extractall(os.path.join(tmp_folder, img_name))

            # rename tar
            # os.rename(os.path.join(path, '{}.tar'.format(item[:-4])), os.path.join(path, '{}.tar'.format(img_name)))

            subfolder = 'DVD_VOL_1'
            for x in os.listdir(os.path.join(tmp_folder, img_name, order_id)):
                if 'DVD_VOL' in x:
                    subfolder = x
                    break

            des_folder = os.path.join(tmp_folder, img_name, order_id, subfolder, order_id)
            # walk through des_folder
            # img_files = []
            # for root, dirs, files in os.walk(des_folder):
            #     img_files.extend([os.path.join(root, x) for x in files
            #                       if img_name in x and (x[-4:] == '.XML' or x[-4:] == '.JPG')])

            rpc_file = os.path.join(des_folder, '{}_PAN'.format(prod_id), '{}.XML'.format(img_name))
            jpg_file = os.path.join(des_folder, '{}_PAN'.format(prod_id), '{}-BROWSE.JPG'.format(img_name))
            img_files = [rpc_file, jpg_file]
            for x in img_files:
                shutil.copy(x, out_folder)

    # remove tmp_folder
    shutil.rmtree(tmp_folder, ignore_errors=True)


if __name__ == '__main__':
    # path = sys.argv[1]
    path = '/home/kai/core3d/wpafb/PAN'
    prep_data(path)
