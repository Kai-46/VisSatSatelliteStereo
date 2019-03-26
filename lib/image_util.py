import os
import shutil

import cv2
import numpy as np
from osgeo import gdal, gdal_array


def parse_proj_str(proj_str):
    idx1 = proj_str.find('UTM zone')
    idx2 = proj_str.find('",')

    sub_str = proj_str[idx1:idx2]
    zone_letter = sub_str[-1]
    zone_number = int(sub_str[-3:-1])

    return zone_number, zone_letter


def stretch_min_max(arr, min=None, max=None, scale=255):
    if min is None:
        min = arr.min()
    if max is None:
        max = arr.max()
    factor = scale / (max - min)
    return np.clip((arr - min) * factor, 0, scale)


def median_std_max(a, std_th=4.5):
    b = a[abs(a - np.nanmedian(a)) < std_th * np.nanstd(a)]
    return np.nanmax(b)


def rgb_to_gray(a):
    return a[:, :, 0] * .299 + a[:, :, 1] * .587 + a[:, :, 2] * .114


def cum_cnt_min_max(a, ll=.02, ul=.98, bins=1024):
    max_value = a.max()
    hist = np.histogram(a, bins=np.arange(1, max_value, max_value / bins))
    counts = hist[0]
    boundary = hist[1]

    min = 0
    max = counts.shape[0]
    sum = counts.sum()
    # TODO - could interpolate the boundary

    # compute lower limit
    ct = 0
    lower = sum * (1.0 - ll)
    for i in range(counts.shape[0] - 1, -1, -1):
        ct = ct + counts[i]
        if ct > lower:
            min = boundary[i]
            break

    # compute upper limit
    ct = 0
    upper = sum * ul
    for i in range(counts.shape[0]):
        ct = ct + counts[i]
        if ct > upper:
            max = boundary[i]
            break
    return min, max


def msi_to_rgb_adjusted(in_file, out_file, bands=None, max_b=None, hist_eq=True):
    if bands is None:
        bands = [5, 3, 2]
    rgb, geo, proj, meta, width, height = read_image(in_file, np_type=np.float32, bands=bands)

    if max_b is None:
        max_b = [0] * len(bands)
        for b in range(len(bands)):
           max_b[b] = median_std_max(rgb[:, :, b])

    for b in range(len(bands)):
        rgb[:, :, b] = stretch_min_max(rgb[:, :, b], 0, max_b[b])

    rgb = rgb.astype(np.uint8)
    if hist_eq:
        ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCR_CB)
        channels = cv2.split(ycrcb)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
        channels[0] = clahe.apply(channels[0])
        rgb = cv2.cvtColor(cv2.merge(channels), cv2.COLOR_YCR_CB2RGB)

    if out_file.endswith(".jpg"):
        cv2.imwrite(out_file, rgb, [cv2.IMWRITE_JPEG_QUALITY, 100])
    elif out_file.endswith(".png"):
        cv2.imwrite(out_file, rgb)
    else:
        write_image(rgb, out_file, geo, proj, meta)


def bounding_box(file):
    ds = gdal.Open(file)
    if ds is None:
        return None, None, None, None
    geo = ds.GetGeoTransform()
    ulx = geo[0]
    uly = geo[3]
    lrx = ulx + ds.RasterXSize * geo[1]
    lry = uly + ds.RasterYSize * geo[5]
    del ds
    return ulx, uly, lrx, lry


def read_info(file):
    ds = gdal.Open(file)
    if ds is None:
        return None, None, None, None, None
    geo = ds.GetGeoTransform()
    proj = ds.GetProjection()
    meta = ds.GetMetadata()
    width = ds.RasterXSize
    height = ds.RasterYSize
    del ds
    return geo, proj, meta, width, height


def read_image(file, np_type=None, bands=None):
    ds = gdal.Open(file)
    geo = ds.GetGeoTransform()
    proj = ds.GetProjection()
    meta = ds.GetMetadata()
    width = ds.RasterXSize
    height = ds.RasterYSize

    if np_type is None:
        # load type based on raster
        np_type = gdal_array.GDALTypeCodeToNumericTypeCode(ds.GetRasterBand(1).DataType)

    # Allocate our numpy array
    if bands is not None:
        num_bands = len(bands)
    else:
        num_bands = ds.RasterCount

    image = np.zeros((ds.RasterYSize, ds.RasterXSize, num_bands), dtype=np_type)

    if bands is None:
        bands = range(1, ds.RasterCount + 1)

    # Loop over desired bands (zero based)
    ct = 0
    for b in bands:
        band = ds.GetRasterBand(b)  # bands are indexed from 1
        image[:, :, ct] = band.ReadAsArray()
        ct = ct + 1

    del ds
    return image, geo, proj, meta, width, height


def read_bands(file, np_type=None, bands=None):
    ds = gdal.Open(file)

    if np_type is None:
        # load type based on raster
        np_type = gdal_array.GDALTypeCodeToNumericTypeCode(ds.GetRasterBand(1).DataType)

    # Allocate our numpy array
    if bands is not None:
        num_bands = len(bands)
    else:
        num_bands = ds.RasterCount

    image = np.zeros((ds.RasterYSize, ds.RasterXSize, num_bands), dtype=np_type)

    if bands is None:
        bands = range(1, ds.RasterCount + 1)

    # Loop over desired bands (zero based)
    ct = 0
    for b in bands:
        band = ds.GetRasterBand(b)  # bands are indexed from 1
        image[:, :, ct] = band.ReadAsArray()
        ct = ct + 1

    del ds
    return image


def get_driver(file):
    f_ext = os.path.splitext(file)[1]
    for i in range(gdal.GetDriverCount()):
        driver = gdal.GetDriver(i)
        if driver.GetMetadataItem(gdal.DCAP_RASTER):
            d_ext_str = driver.GetMetadataItem(gdal.DMD_EXTENSIONS)
            if d_ext_str is not None:
                for d_ext in d_ext_str.split(' '):
                    if f_ext == '.' + d_ext:
                        return driver
    return None


def write_image(image, out_file, geo=None, proj=None, meta=None, np_type=None, bands=None, options=[], no_data=None):
    driver = get_driver(out_file)
    if np_type is None:
        np_type = np.float32

    if len(image.shape) == 2:
        # This is a single band image - expand the dimension
        image = np.expand_dims(image, axis=2)

    if bands is None:
        bands = range(image.shape[2])

    out = driver.Create(out_file, image.shape[1], image.shape[0], len(bands),
                        gdal_array.NumericTypeCodeToGDALTypeCode(np_type), options=options)

    for b in bands:
        band = out.GetRasterBand(b + 1)
        band.WriteArray(image[..., b].astype(np_type), 0, 0)
        if no_data is not None:
            band.SetNoDataValue(no_data)
        band.FlushCache()

    if geo is not None:
        out.SetGeoTransform(geo)

    if proj is not None:
        out.SetProjection(proj)

    if meta is not None:
        out.SetMetadata(meta)

    del out


def translate_2d_px(in_path, out_path, x, y):
    """
    Translates a geotiff tile by amount trans (in pixels)
    and saves to new geotiff
    """
    shutil.copyfile(in_path, out_path)

    ds = gdal.Open(out_path, gdal.GA_Update)
    x_tl, x_res, dx_dy, y_tl, dy_dx, y_res = ds.GetGeoTransform()

    # compute shift of 1 pixel RIGHT in X direction
    shift_x = x * x_res
    # compute shift of 2 pixels UP in Y direction
    # y_res likely negative, because Y decreases with increasing Y index
    shift_y = y * y_res

    # assign new geo transform
    ds.SetGeoTransform((x_tl + shift_x, x_res, dx_dy, y_tl + shift_y, dy_dx, y_res))
    # ensure changes are committed
    ds.FlushCache()
    del ds
    return shift_x, shift_y


def translate_3d(in_path, out_path, x, y, z=0.0):
    """
    Translates a geotiff tile by amount trans (in meters)
    and saves to new geotiff
    """
    shutil.copyfile(in_path, out_path)

    ds = gdal.Open(out_path, gdal.GA_Update)
    x_tl, x_res, dx_dy, y_tl, dy_dx, y_res = ds.GetGeoTransform()

    # assign new geo transform
    ds.SetGeoTransform((x_tl + x, x_res, dx_dy, y_tl + y, dy_dx, y_res))

    if z != 0.0:
        for b in range(1, ds.RasterCount + 1):
            band = ds.GetRasterBand(b)  # bands are indexed from 1
            arr = band.ReadAsArray()
            arr[~np.isnan(arr)] += z
            band.WriteArray(arr)

    # ensure changes are committed
    ds.FlushCache()
    del ds


# NOTE THESE METHODS WERE FROM THE METRICS CODE BUT IS ADAPTED FOR MULTIPLE BANDS

def getMetadata(inputinfo):

    # dataset input
    if isinstance(inputinfo,gdal.Dataset):
        dataset = inputinfo
        FLAG_CLOSE = False

    # file input
    elif isinstance(inputinfo,str):
        filename = inputinfo
        if not os.path.isfile(filename):
            raise IOError('Cannot locate file <{}>'.format(filename))

        dataset = gdal.Open(filename, gdal.GA_ReadOnly)
        FLAG_CLOSE = True

    # unrecognized input
    else:
        raise IOError('Unrecognized getMetadata input')

    # read metadata
    meta = {
        'RasterXSize':  dataset.RasterXSize,
        'RasterYSize':  dataset.RasterYSize,
        'RasterCount':  dataset.RasterCount,
        'Projection':   dataset.GetProjection(),
        'GeoTransform': list(dataset.GetGeoTransform()),
    }

    # cleanuo
    if FLAG_CLOSE: dataset = None
    return meta

def imageWarp(file_src: str, file_dst: str, offset=None, interp_method: int = gdal.gdalconst.GRA_Bilinear, noDataValue=None):

    # destination metadata
    meta_dst = getMetadata(file_dst)

    # GDAL memory driver
    mem_drv = gdal.GetDriverByName('MEM')

    # copy source to memory
    tmp = gdal.Open(file_src, gdal.GA_ReadOnly)
    dataset_src = mem_drv.CreateCopy('',tmp)
    tmp = None

    # change no data value to new "noDataValue" input if necessary,
    # making sure to adjust the underlying pixel values
    band = dataset_src.GetRasterBand(1)
    NDV = band.GetNoDataValue()

    if noDataValue is not None and noDataValue != NDV:
        if NDV is not None:
            img = band.ReadAsArray()
            img[img==NDV] = noDataValue
            band.WriteArray(img)
        band.SetNoDataValue(noDataValue)
        NDV = noDataValue

    # source metadata
    meta_src = getMetadata(dataset_src)

    # Apply registration offset
    if offset is not None:

        # offset error: offset is defined in destination projection space,
        # and cannot be applied if source and destination projections differ
        if meta_src['Projection'] != meta_dst['Projection']:
            print('IMAGE PROJECTION\n{}'.format(meta_src['Projection']))
            print('OFFSET PROJECTION\n{}'.format(meta_dst['Projection']))
            raise ValueError('Image/Offset projection mismatch')

        transform = meta_src['GeoTransform']
        transform[0] += offset[0]
        transform[3] += offset[1]
        dataset_src.SetGeoTransform(transform)


    # no reprojection necessary
    if meta_src == meta_dst:
        print('  No reprojection')
        dataset_dst = dataset_src

    # reprojection
    else:

        # file, xsz, ysz, nbands, dtype
        dataset_dst = mem_drv.Create('', meta_dst['RasterXSize'], meta_dst['RasterYSize'],
            meta_src['RasterCount'], gdal.GDT_Float32)

        dataset_dst.SetProjection(meta_dst['Projection'])
        dataset_dst.SetGeoTransform(meta_dst['GeoTransform'])

        if NDV is not None:
            band = dataset_dst.GetRasterBand(1)
            band.SetNoDataValue(NDV)
            band.Fill(NDV)

        # input, output, inputproj, outputproj, interp
        gdal.ReprojectImage(dataset_src, dataset_dst, meta_src['Projection'],
             meta_dst['Projection'], interp_method)

    image = np.zeros((meta_dst['RasterYSize'], meta_dst['RasterXSize'], meta_src['RasterCount']), dtype=np.float32)

    for b in range(1, meta_src['RasterCount'] + 1):
        band = dataset_dst.GetRasterBand(b)  # bands are indexed from 1
        image[:, :, b-1] = band.ReadAsArray()
    return image


if __name__ == "__main__":
    gt_tif = '/data2/kz298/mvs3dm_result/MasterSequesteredPark/evaluation/eval_ground_truth.tif'
    image, geo, proj, meta, width, height = read_image(gt_tif)

    zone_number, zone_letter = parse_proj_str(proj)

    print('hello')
