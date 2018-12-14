# extract tiles

import os
from lib.rpc_model import RPCModel
from lib.parse_meta import parse_meta
from lib.gen_grid import gen_grid
from lib.solve_perspective import solve_perspective
import utm
import numpy as np
import shutil
import imageio
import quaternion
import json
import cv2
from lib import clean_data
import warnings
import xml.etree.ElementTree as ET
# from lxml import etree
import unicodedata

# import matplotlib.pyplot as plt

# pinhole approx and undistort the image
# in_folder is the folder for perspective cameras
def pinhole_approx(in_folder, out_folder):
    out_folder = os.path.abspath(out_folder)
    if os.path.exists(out_folder):
        shutil.rmtree(out_folder, ignore_errors=True)
    logging.info('in_folder: {}'.format(in_folder))
    logging.info('out_folder: {}'.format(out_folder))
    os.mkdir(out_folder)
    os.mkdir(os.path.join(out_folder, 'images'))
    os.mkdir(os.path.join(out_folder, 'images_uncrop'))

    cameras_line_format = '{camera_id} PINHOLE {width} {height} {fx} {fy} {cx} {cy}\n'
    with open(os.path.join(in_folder, 'camera_data.json')) as fp:
        camera_dict = json.load(fp)

    skew_dict = {}

    for img_name in camera_dict:
        params = camera_dict[img_name][0].strip().split(' ')
        width = int(params[2])
        height = int(params[3])
        fx = float(params[4])
        fy = float(params[5])
        cx = float(params[6])
        cy = float(params[7])
        s = float(params[8])
        # compute homography
        norm_skew = s / fy

        skew_dict[img_name] = norm_skew

        logging.info('removing skew, image: {}, normalized skew: {}'.format(img_name, norm_skew))
        homography = np.array([[1, -norm_skew, 0],
                               [0, 1, 0],
                               [0, 0, 1]])
        # compute bounding box after applying the above homography
        points = np.dot(homography, np.array([[0., width, width, 0.],
                                              [0., 0., height, height],
                                              [1., 1., 1., 1.]]))
        w = int(np.min((points[0, 1], points[0, 2])))
        h = int(np.min((points[1, 2], points[1, 3])))

        logging.info('original image size, width: {}, height: {}'.format(width, height))
        logging.info('corrected image size, width: {}, height: {}'.format(w, h))
        im_src = imageio.imread(os.path.join(in_folder, 'images', img_name)).astype(dtype=np.float64)

        # note that opencv axis direction might be (row, col)???????

        img_dst = cv2.warpPerspective(im_src, homography, (w, h))
        imageio.imwrite(os.path.join(out_folder, 'images', img_name), img_dst.astype(dtype=np.uint8))

        # also save the uncrop version for diagnosis
        w_big = int(np.max((points[0, 1], points[0, 2]-points[0, 3])))
        h_big = int(np.max((points[1, 2], points[1, 3])))
        img_uncrop = cv2.warpPerspective(im_src, homography, (w_big, h_big))
        imageio.imwrite(os.path.join(out_folder, 'images_uncrop', img_name), img_uncrop.astype(dtype=np.uint8))

        camera_line = cameras_line_format.format(camera_id="{camera_id}", width=w, height=h,
                                                 fx=fx, fy=fy, cx=cx - s * cy / fy, cy=cy)
        camera_dict[img_name][0] = camera_line

        logging.info('\n\n')

    with open(os.path.join(out_folder, 'camera_data.json'), 'w') as fp:
        json.dump(camera_dict, fp, indent=2)

    with open(os.path.join(out_folder, 'skews.json'), 'w') as fp:
        json.dump(skew_dict, fp, indent=2)

    shutil.copy(os.path.join(in_folder, 'roi.json'), out_folder)
    shutil.copytree(os.path.join(in_folder, 'metas'), os.path.join(out_folder, 'metas'))

class TileExtractor(object):
    def __init__(self, path):
        self.path = path

        self.ntf_list = []
        self.xml_list = []
        clean_data.prep_data(path)
        cleaned_data = os.path.join(path, 'cleaned_data')
        for item in sorted(os.listdir(cleaned_data)):
            if item[-4:] == '.NTF':
                xml_file = os.path.join(cleaned_data, '{}.XML'.format(item[:-4]))
                self.ntf_list.append(os.path.join(cleaned_data, item))
                self.xml_list.append(xml_file)
        self.meta_dicts = [parse_meta(xml_file) for xml_file in self.xml_list]
        self.rpc_models = [RPCModel(meta_dict) for meta_dict in self.meta_dicts]

        self.cnt = len(self.ntf_list)
        # create a ascending date index
        tmp = [(i, self.meta_dicts[i]['capTime']) for i in range(self.cnt)]
        tmp = sorted(tmp, key=lambda x: x[1])
        self.time_index = [x[0] for x in tmp]

        self.min_height, self.max_height = self.height_range()

        logging.info('min_height, max_height: {}, {}'.format(self.min_height, self.max_height))

    def height_range(self):
        z_min = -1e10
        z_max = 1e10
        for i in range(self.cnt):
            z_min_candi = self.rpc_models[i].altOff - 0. * self.rpc_models[i].altScale
            z_max_candi = self.rpc_models[i].altOff + 0.7 * self.rpc_models[i].altScale
            if z_min_candi > z_min:
                z_min = z_min_candi
            if z_max_candi < z_max:
                z_max = z_max_candi
        return z_min, z_max

    def extract_roi_utm(self, zone_number, zone_letter, ul_east, ul_north, lr_east, lr_north, out_folder):
        out_folder = os.path.abspath(out_folder)
        if os.path.exists(out_folder):
            shutil.rmtree(out_folder, ignore_errors=True)
        logging.info('out_folder: {}'.format(out_folder))
        os.mkdir(out_folder)
        os.mkdir(os.path.join(out_folder, 'images'))
        os.mkdir(os.path.join(out_folder, 'metas'))

        # write to roi.json
        roi_dict = { 'zone_number': zone_number,
                     'zone_letter': zone_letter,
                     'x': ul_east,
                     'y': ul_north,
                     'w': lr_east - ul_east,
                     'h': ul_north - lr_north }
        with open(os.path.join(out_folder, 'roi.json'), 'w') as fp:
            json.dump(roi_dict, fp, indent=2)

        ul_lat, ul_lon = utm.to_latlon(ul_east, ul_north, zone_number, zone_letter)
        lr_lat, lr_lon = utm.to_latlon(lr_east, lr_north, zone_number, zone_letter)

        cameras_line_format = '{camera_id} PERSPECTIVE {width} {height} {fx} {fy} {cx} {cy} {s}\n'
        images_line_format = '{image_id} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {camera_id} {image_name}\n\n'
        camera_dict = {}

        # create lat_lon_height grid
        lat_points = np.linspace(ul_lat, lr_lat, 20)
        lon_points = np.linspace(ul_lon, lr_lon, 20)
        z_points = np.linspace(self.min_height, self.max_height, 20)

        xx_lat, yy_lon, zz = gen_grid(lat_points, lon_points, z_points)

        # create north_east_height grid
        north_points = np.linspace(ul_north, lr_north, 20)
        east_points = np.linspace(ul_east, lr_east, 20)

        xx, yy, zz = gen_grid(north_points, east_points, z_points)

        # use a smaller number and change to the right-handed coordinate frame
        xx = ul_north - xx
        yy = yy - ul_east

        useful_cnt = 0 # number of useful images
        for k in range(self.cnt):
            # debug
            # if k==0:
            #     continue
            #k = 1 # debug

            logging.info('processing image {}/{}, already collect {} useful images...'.format(k, self.cnt, useful_cnt))

            i = self.time_index[k]   # image index

            col, row = self.rpc_models[i].projection(xx_lat, yy_lon, zz)

            # compute the bounding box
            ul_row = int(np.round(np.min(row)))
            ul_col = int(np.round(np.min(col)))
            width = int(np.round(np.max(col))) - ul_col
            height = int(np.round(np.max(row))) - ul_row

            # cut image
            in_ntf = self.ntf_list[i]
            cap_time = self.meta_dicts[i]['capTime'].strftime("%Y%m%d%H%M%S")
            ntf_width = self.meta_dicts[i]['width']
            ntf_height = self.meta_dicts[i]['height']

            # check if the bounding box goes out of the image
            keep, bbx = TileExtractor.check_bbx((ntf_width, ntf_height), (ul_col, ul_row, width, height))
            if not keep:
                logging.warning('will discard this image due to small coverage of target area, ntf: {}'.format(in_ntf))
                continue
            else:
                ul_col, ul_row, width, height = bbx

            out_png = os.path.join(out_folder, 'images/{:03d}_{}.png'.format(useful_cnt, cap_time))
            TileExtractor.cut_image(in_ntf, out_png, (ntf_width, ntf_height), (ul_col, ul_row, width, height))

            # approximate the camera
            # try remove the outliers in order to improve approximation accuracy
            keep_mask = np.logical_and(row >= ul_row, col >=  ul_col)
            keep_mask = np.logical_and(keep_mask, row < ul_row + height)
            keep_mask = np.logical_and(keep_mask, col < ul_col + width)

            # subtract the cutting offset
            row -= ul_row
            col -= ul_col

            # modify RPC camera parameters
            meta_dict = self.meta_dicts[i]
            rpc_dict = meta_dict['rpc']
            # also subtract the cutting offset here
            rpc_dict['rowOff'] = rpc_dict['rowOff'] - ul_row
            rpc_dict['colOff'] = rpc_dict['colOff'] - ul_col

            meta_dict['rpc'] = rpc_dict
            meta_dict['capTime'] = meta_dict['capTime'].isoformat()
            with open(os.path.join(out_folder, 'metas/{:03d}_{}.json'.format(useful_cnt, cap_time)), 'w') as fp:
                json.dump(meta_dict, fp, indent=2)

            r, q, t, _ = solve_perspective(xx, yy, zz, col, row, keep_mask)

            # write to colmap format
            img_name = out_png[out_png.rfind('/')+1:]
            logging.info('write to camera dict: {}'.format(img_name))
            camera_line = cameras_line_format.format(camera_id="{camera_id}", width=width, height=height,
                                                     fx=r[0, 0], fy=r[1, 1], cx=r[0, 2], cy=r[1, 2], s=r[0, 1])
            quat = quaternion.from_rotation_matrix(q)
            image_line = images_line_format.format(image_id="{image_id}", qw=quat.w, qx=quat.x, qy=quat.y, qz=quat.z,
                                                   tx=t[0, 0], ty=t[1, 0], tz=t[2, 0], camera_id="{camera_id}",
                                                   image_name=img_name)
            camera_dict[img_name] = (camera_line, image_line)

            # increase number of useful images
            useful_cnt += 1

            logging.info('\n\n')

        with open(os.path.join(out_folder, 'camera_data.json'), 'w') as fp:
            json.dump(camera_dict, fp, indent=2)

    def extract_roi_latlon(self, ul_lat, ul_lon, w, h, out_folder):
        (ul_east, ul_north, zone_number, zone_letter) = utm.from_latlon(ul_lat, ul_lon)
        self.extract_roi_utm(zone_number, zone_letter, ul_east, ul_north, ul_east + w, ul_north - h, out_folder)

    @classmethod
    def has_cloud(cls, rpc_file):
        # utf-8 encoding error
        with open(rpc_file, encoding='utf-8', errors='ignore') as fp:
            content = fp.read()
        if content.find('UTF-8') < 0:
            logging.warning('will discard this rpc file because it is not encoded in utf-8: {}'.format(rpc_file))
            return True

        # remove control characters
        content = "".join([ch for ch in content if unicodedata.category(ch)[0] != "C"])

        with open(rpc_file, 'w') as fp:
            fp.write(content)
        tree = ET.parse(rpc_file)
        root = tree.getroot()
        has_cloud = root.find('IMD/IMAGE/CLOUDCOVER').text
        has_cloud = float(has_cloud)
        thres = 0.5
        if has_cloud > thres:
            logging.warning('will discard this image because of too many clouds, has_cloud: {}, rpc: {}'.format(has_cloud, rpc_file))
            return True
        return False

    @classmethod
    def check_bbx(cls, ntf_size, bbx_size):
        keep = True

        (ntf_width, ntf_height) = ntf_size
        (ul_col, ul_row, width, height) = bbx_size
        lr_col = ul_col + width - 1
        lr_row = ul_row + height - 1

        # check if the bounding box goes out of the image
        if ul_col < 0 or ul_row < 0 or lr_col >= ntf_width or lr_row >= ntf_height:
            logging.warning('bounding box out of content, ntf_width, ntf_height: {}, {}, bounding box: {}, {}, {}, {}'
                          .format(ntf_width, ntf_height, ul_col, ul_row, width, height))
            if  ul_col >= ntf_width or lr_col < 0 or ul_row >= ntf_height or lr_row < 0:
                logging.warning('bounding box completely out of content')
                keep = False
            else:
                # compute how much portion that goes out of content
                inside_ul_col = ul_col if ul_col >= 0 else 0
                inside_ul_row = ul_row if ul_row >= 0 else 0
                inside_lr_col = lr_col if lr_col < ntf_width else ntf_width - 1
                inside_lr_row = lr_row if lr_row < ntf_height else ntf_height - 1

                inside_width = inside_lr_col - inside_ul_col + 1
                inside_height = inside_lr_row - inside_ul_row + 1
                overlap_ratio = float( inside_width * inside_height) / (width * height)

                logging.info('{} of bounding box out of content'.format(1 - overlap_ratio))

                if overlap_ratio < 0.5:
                    keep = False
                else:
                    logging.info('fixing bounding box, previous: {}, {}, {}, {}, fixed: {}, {}, {}, {}'.format(
                        ul_col, ul_row, width, height, inside_ul_col, inside_ul_row, inside_width, inside_height
                    ))
                    bbx_size = (inside_ul_col, inside_ul_row, inside_width, inside_height)

        if keep:
            return (keep, bbx_size)
        else:
            return (keep, None)

    @classmethod
    def cut_image(cls, in_ntf, out_png, ntf_size, bbx_size):
        (ntf_width, ntf_height) = ntf_size
        (ul_col, ul_row, width, height) = bbx_size

        # assert bounding box is completely inside the image
        assert(ul_col + width - 1 < ntf_width and ul_row + height - 1 < ntf_height)

        logging.info('ntf image to cut: {}, width, height: {}, {}'.format(in_ntf, ntf_width, ntf_height))
        logging.info('cut image bounding box, ul_col, ul_row, width, height: {}, {}, {}, {}'.format(ul_col, ul_row, width, height))
        logging.info('png image to save: {}'.format(out_png))

        # note the coordinate system of .ntf
        cmd = 'gdal_translate -of png -ot UInt16 -srcwin {} {} {} {} {} {}'\
            .format(ul_col, ul_row, width, height, in_ntf, out_png)
        os.system(cmd)
        os.remove('{}.aux.xml'.format(out_png))

        # scale to [0, 255]
        im = imageio.imread(out_png).astype(dtype=np.float64)
        # h, w = im.shape
        # if w != width or h != height:
        #     logging.warning('cut image size is not what is wanted, discarding: {}'.format(in_ntf))
        #     os.remove(out_png)
        #     return False

        # tmp = im.reshape((-1, 1))
        # tmp = (tmp - np.min(tmp)) / (np.max(tmp) - np.min(tmp))
        # plt.hist(tmp, bins=100, density=True, cumulative=True)
        # plt.xlabel('normalized intensity')
        # plt.title('original image intensity')
        # plt.show()

        im = np.power(im, 1.0 / 2.2)    # non-uniform sampling

        # tmp = im.reshape((-1, 1))
        # tmp = (tmp - np.min(tmp)) / (np.max(tmp) - np.min(tmp))
        # plt.hist(tmp, bins=100, density=True, cumulative=True)
        # plt.xlabel('normalized intensity')
        # plt.title('after applying the non-linear encoding')
        # plt.show()

        below_thres = np.percentile(im.reshape((-1, 1)), 1)
        im[im < below_thres] = below_thres
        # cut off the big values
        above_thres = np.percentile(im.reshape((-1, 1)), 99.5)
        im[im > above_thres] = above_thres
        im = 255 * (im - below_thres) / (above_thres - below_thres)

        # tmp = im.reshape((-1, 1))
        # tmp = (tmp - np.min(tmp)) / (np.max(tmp) - np.min(tmp))
        # plt.hist(tmp, bins=100, density=True, cumulative=True)
        # plt.xlabel('normalized intensity')
        # plt.title('after applying the non-linear encoding')
        # plt.show()

        # remove the unneeded one
        os.remove(out_png)

        imageio.imwrite(out_png, im.astype(dtype=np.uint8))

        # return True

def extract_aoi(aoi_name, data_name):
    logging.info('aoi_name: {}, data_name: {}'.format(aoi_name, data_name))

    with open('aoi_utm.json') as fp:
        roi = json.load(fp)[aoi_name]

    out_folder = '/data2/kz298/core3d_aoi/{}'.format(aoi_name)
    if os.path.exists(out_folder):
        shutil.rmtree(out_folder, ignore_errors=True)
    os.mkdir(out_folder)

    path = '/data2/kz298/core3d_pan/{}'
    tile_extractor = TileExtractor(path.format(data_name))
    tile_extractor.extract_roi_utm(roi['utm_band'], roi['hemisphere'],
                                   roi['x'], roi['y'], roi['x'] + roi['w'], roi['y'] - roi['h'], out_folder)
    pinhole_approx(out_folder, '{}_pinhole'.format(out_folder))

    # shutil.rmtree(out_folder, ignore_errors=True)


def test():
    path = '/home/kai/core3d/jacksonville/PAN'
    tile_extractor = TileExtractor(path)


if __name__ == '__main__':
    # test()

    # extract_aoi('wpafb')

    #extract_aoi(sys.argv[1], sys.argv[2])

    extract_aoi('aoi-d1-wpafb', 'wpafb')
    extract_aoi('aoi-d2-wpafb', 'wpafb')
    extract_aoi('aoi-d3-ucsd', 'ucsd')
    extract_aoi('aoi-d4-jacksonville', 'jacksonville')

    # path = '/home/kai/satellite_project/dataset/core3d/PAN/jacksonville'
    # tile_extractor = TileExtractor(path)

    # out_folder = 'data_roi_1000'
    # tile_extractor.extract_roi_utm(17, 'N', 435532.000, 3354107.000, 435532.000+1385, 3354107.000-1413, out_folder)
    # pinhole_approx(out_folder, '{}_pinhole'.format(out_folder))

    # out_folder = 'data_100'
    # tile_extractor.extract_roi_utm(17, 'N', 435532.000+700, 3354107.000, 435532.000 + 700 + 100, 3354107.000 - 100, out_folder)

    # out_folder = 'data_500'
    # tile_extractor.extract_roi_utm(17, 'N', 435532.000+700, 3354107.000, 435532.000 + 700 + 500, 3354107.000 - 500, out_folder)
    # pinhole_approx(out_folder, '{}_pinhole'.format(out_folder))

    # out_folder = 'data_200'
    # tile_extractor.extract_roi_utm(17, 'N', 435532.000+700, 3354107.000, 435532.000 + 700 + 200, 3354107.000 - 200, out_folder)
    # pinhole_approx(out_folder, '{}_pinhole'.format(out_folder))


