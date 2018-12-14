import os
import tarfile
import shutil
import unicodedata


# first find .NTF file, and extract order_id, prod_id, standard name
# then extract rpc file and preview image from the .tar file

def clean_data(dataset_dir, out_dir):
    assert(os.path.exists(out_dir))

    dataset_dir = os.path.abspath(dataset_dir)
    print('dataset path: {}'.format(dataset_dir))
    print('will save files to folder: {}'.format(out_dir))
    print('the standard format is: <7 char date><6 char time>-P1BS-<20 char product id>.NTF\n\n')

    # if os.path.exists(out_dir):
    #     shutil.rmtree(out_dir, ignore_errors=True)
    # os.mkdir(out_dir)

    tmp_dir = os.path.join(out_dir, 'tmp')
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir, ignore_errors=True)
    os.mkdir(tmp_dir)

    for item in sorted(os.listdir(dataset_dir)):
        if item[-4:] == '.NTF':
            # get order_id, prod_id
            idx = item.find('-P1BS-')
            order_id = item[idx+6:idx+21]
            prod_id = item[idx+6:idx+26]
            img_name = item[idx-13:idx+26]

            # rename NTF
            # os.rename(os.path.join(dataset_dir, item), os.path.join(dataset_dir, '{}.NTF'.format(img_name)))

            os.link(os.path.join(dataset_dir, item), os.path.join(out_dir, '{}.NTF'.format(img_name)))

            tar = tarfile.open(os.path.join(dataset_dir, '{}.tar'.format(item[:-4])))

            tar.extractall(os.path.join(tmp_dir, img_name))

            # rename tar
            # os.rename(os.path.join(dataset_dir, '{}.tar'.format(item[:-4])), os.path.join(dataset_dir, '{}.tar'.format(img_name)))

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

    # remove tmp_dir
    shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == '__main__':
    # path = sys.argv[1]
    # path = '/home/kai/core3d/wpafb/PAN'
    # path = '/data2/kz298/dataset/core3d/performer_source_data/jacksonville/satellite_imagery/WV3/PAN'
    # path = '/data2/kz298/dataset/core3d/performer_source_data/san_fernando/satellite_imagery/WV3/PAN'
    # path = '/data2/kz298/dataset/core3d/performer_source_data/ucsd/satellite_imagery/WV3/PAN'
    path = '/data2/kz298/dataset/core3d/performer_source_data/wpafb/satellite_imagery/WV3/PAN'
    clean_data(path)
