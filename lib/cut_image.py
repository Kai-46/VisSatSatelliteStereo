import os
import tempfile
import cv2
import numpy as np
from osgeo import gdal, gdal_array
from skimage import exposure
from lib.run_cmd import run_cmd
import logging

try:
    from s2p import rpc_utils                    
except ImportError:
    print('Not supporting Pan-sharpening')


def load_rpc(file):
    rpc_file = tempfile.mktemp('.rpc')
    rpc = rpc_utils.rpc_from_geotiff(file, rpc_file)
    os.remove(rpc_file)
    return rpc


def apply_rpc(ntf, out_file):
    logging.info("Applying RPC: {}".format(ntf))
    cmd = "gdalwarp -q -overwrite -multi -wo NUM_THREADS=ALL_CPUS -r cubicspline -rpc " \
          "-co BIGTIFF=IF_SAFER -dstnodata 0 -srcnodata 0 -wo INIT_DEST=0 {} {}" \
        .format(ntf, out_file)
    run_cmd(cmd)


def crop_roi(ntf, ul_col, ul_row, width, height, crop_out):
    logging.info("Cropping {} to ROI".format(ntf))
    run_cmd('gdal_translate -q -srcwin %d %d %d %d "%s" "%s"' % (ul_col, ul_row, width, height, ntf, crop_out))


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


def write_image(image, out_file, geo=None, proj=None, meta=None, np_type=None, bands=None, options=None, no_data=None):
    driver = get_driver(out_file)
    if np_type is None:
        if image.dtype == np.float64:
            np_type = np.float32
        elif image.dtype == np.bool_:
            np_type = np.uint8
        else:
            np_type = image.dtype

    if options is None:
        options = []

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


def read_image(file, np_type=None, bands=None, squeeze=False):
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
    if squeeze:
        image = np.squeeze(image)
    return image, geo, proj, meta, width, height


def to_rgb_adjusted_new(rgb_in_name, rgb_out_name, percentiles=None, bands=None, hist_eq=True, nodata=None):
    if percentiles is None:
        percentiles = [1, 99]
    if bands is None:
        bands = [1, 2, 3]
    rgb, geo, proj, meta, width, height = read_image(rgb_in_name, np_type=np.float32, bands=bands)
    flat_rgb = rgb.reshape(rgb.shape[0] * rgb.shape[1], rgb.shape[2])
    mask = None

    if nodata is not None:
        lo_range, hi_range = np.zeros(rgb.shape[2]), np.zeros(rgb.shape[2])
        for i in range(rgb.shape[2]):
            lo_range[i], hi_range[i] = np.percentile(flat_rgb[flat_rgb[..., i] != nodata, i], percentiles, axis=0)
    else:
        lo_range, hi_range = np.percentile(flat_rgb, percentiles, axis=0)

    rgb = np.maximum(np.minimum(rgb, hi_range), lo_range)
    if hist_eq:
        for i in range(rgb.shape[2]):
            rgb[..., i] = exposure.rescale_intensity(rgb[..., i], out_range=(0, 255))
        rgb = rgb.astype(np.uint8)
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
        lab[..., 0] = clahe.apply(lab[..., 0])
        rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    else:
        for i in range(rgb.shape[2]):
            rgb[..., i] = exposure.rescale_intensity(rgb[..., i], out_range=(0, 255))
        rgb = rgb.astype(np.uint8)

    if mask is not None:
        rgb[mask] = nodata

    if rgb_out_name.endswith(".jpg"):
        cv2.imwrite(rgb_out_name, rgb, [cv2.IMWRITE_JPEG_QUALITY, 100])
    elif rgb_out_name.endswith(".png"):
        cv2.imwrite(rgb_out_name, rgb)
    else:
        write_image(rgb, rgb_out_name, geo, proj, meta, np_type=np.uint8)


pan_sharpen_weights = {

    'WV02_RGB': [0.00, 0.25, 0.23, 0.00, 0.53, 0.00, 0.00, 0.00],
    'WV03_RGB': [0.00, 0.25, 0.23, 0.00, 0.53, 0.00, 0.00, 0.00],
    'GE01_RGB': [0.25, 0.23, 0.53, 0.00],

    'WV02_MS': [0.005, 0.142, 0.209, 0.144, 0.234, 0.157, 0.116, 0.000],  # MSI
    'WV03_MS': [0.005, 0.142, 0.209, 0.144, 0.234, 0.157, 0.116, 0.000],  # MSI
    'GE01_MS': [0.142, 0.209, 0.234, 0.116],  # MSI
}
gdal_bands = {
    'WV02_MS': {'C': 1, 'B': 2, 'G': 3, 'Y': 4, 'R': 5, 'RE': 6, 'N1': 7, 'N2': 8},
    'WV03_MS': {'C': 1, 'B': 2, 'G': 3, 'Y': 4, 'R': 5, 'RE': 6, 'N1': 7, 'N2': 8},
    'GE01_MS': {'B': 1, 'G': 2, 'R': 3, 'N': 4}
}


def get_sat_key(meta):
    return meta['NITF_PIAIMC_SENSNAME'] + '_' + meta['NITF_ICAT']


def pan_sharpen(pan_name, msi_name, out_name, weights, bands=None):
    if bands is None:
        bands = [x+1 for x in range(len(weights))]

    cmd = 'gdal_pansharpen.py -r cubicspline -q -threads ALL_CPUS -co BIGTIFF=YES '
    cmd_b = ' {}'.format(pan_name)
    for b in bands:
        cmd = cmd + " -w {}".format(weights[b-1])
        cmd_b = cmd_b + " {},band={}".format(msi_name, b)
    run_cmd(cmd + cmd_b + ' ' + out_name)


def pan_sharpen_unweighted(pan_name, msi_name, out_name):
    cmd = 'gdal_pansharpen.py -r cubicspline -q -threads ALL_CPUS -co BIGTIFF=YES {} {} {}'.format(pan_name, msi_name, out_name)
    os.system(cmd)


def create_rgb_aligned_with_pan(pan_ntf, x, y, w, h, msi_ntf, output_png):
    pan_crop_out = tempfile.mktemp(".tif")
    msi_crop_out = tempfile.mktemp(".tif")
    pan_rpc_applied = tempfile.mktemp(".tif")
    msi_rpc_applied = tempfile.mktemp(".tif")
    pan_sharpened_rgb_equalized = tempfile.mktemp(".tif")
    pan_sharpened_rgb = tempfile.mktemp(".tif")

    pan_rpc = load_rpc(pan_ntf)
    crop_roi(pan_ntf, x, y, w, h, pan_crop_out)

    msi_rpc = load_rpc(msi_ntf)
    x, y, w, h = rpc_utils.corresponding_roi(pan_rpc, msi_rpc, x, y, w, h)
    crop_roi(msi_ntf, x, y, w, h, msi_crop_out)

    apply_rpc(pan_crop_out, pan_rpc_applied)
    apply_rpc(msi_crop_out, msi_rpc_applied)
    _, _, msi_meta, _, _ = read_info(msi_crop_out)

    logging.info("Pan-sharpening image...")
    # pan sharpen the using the pan and msi data that are now aligned by their rpc
    weights = pan_sharpen_weights[get_sat_key(msi_meta)]
    b = gdal_bands[get_sat_key(msi_meta)]

    # pan_sharpened_ps = os.path.join('/home/wdixon/jacksonville/', os.path.basename(output_png).replace(".png", "_ps.tif"))
    # pan_sharpen_unweighted(pan_rpc_applied, msi_rpc_applied, pan_sharpened_ps)

    pan_sharpen(pan_rpc_applied, msi_rpc_applied, pan_sharpened_rgb, weights, bands=[b['R'], b['G'], b['B']])
    to_rgb_adjusted_new(pan_sharpened_rgb, pan_sharpened_rgb_equalized, hist_eq=True, nodata=0)

    logging.info("Mapping color space...")
    # setup the rpc transformers
    ds = gdal.Open(pan_crop_out, gdal.GA_ReadOnly)

    # map pixel locations in the pan ntf to lon/lat coordinates
    ntf_rr, ntf_cc = np.meshgrid(np.arange(0, ds.RasterYSize), np.arange(0, ds.RasterXSize))
    ntf_cc_rr = np.vstack((ntf_cc.ravel(), ntf_rr.ravel())).T

    # use the rpc to get all the pixel coordinates at lon/lat
    ntf_xf = gdal.Transformer(ds, None, ['METHOD=RPC'])
    ntf_lon_lat, ntf_lon_lat_success = ntf_xf.TransformPoints(0, ntf_cc_rr)

    # convert the utm coordinates to the pixel coordinates in the clr image
    rgb_ds = gdal.Open(pan_sharpened_rgb_equalized)
    rgb_xf = gdal.Transformer(rgb_ds, None, [])
    rgb_cc_rr_f, rgb_cc_rr_success = rgb_xf.TransformPoints(1, ntf_lon_lat)
    rgb_cc_rr_f = np.array(rgb_cc_rr_f)

    rgb, *_ = read_image(pan_sharpened_rgb_equalized)
    idx = np.where(np.logical_and(rgb_cc_rr_success, np.logical_and(
        np.logical_and(rgb_cc_rr_f[:, 1] >= 0, rgb_cc_rr_f[:, 1] < rgb.shape[0]),
        np.logical_and(rgb_cc_rr_f[:, 0] >= 0, rgb_cc_rr_f[:, 0] < rgb.shape[1]))))

    # create map of the indices that are in bounds
    rgb_cc_rr_map = rgb_cc_rr_f[idx].astype(np.float32)
    xy_map = np.zeros((ds.RasterYSize, ds.RasterXSize, 2), dtype=np.float32)
    xy_map[ntf_rr.ravel()[idx], ntf_cc.ravel()[idx]] = rgb_cc_rr_map[:, :2]
    c_pan = cv2.remap(rgb, xy_map, None, cv2.INTER_LANCZOS4)

    # write out color image in the pixel space of the pan image
    logging.info("Writing out color image...")
    bgr = cv2.cvtColor(c_pan, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_png, bgr)

    del rgb_ds
    del ds

    os.remove(pan_crop_out)
    os.remove(msi_crop_out)
    os.remove(pan_rpc_applied)
    os.remove(msi_rpc_applied)
    os.remove(pan_sharpened_rgb_equalized)
    os.remove(pan_sharpened_rgb)


def cut_image(in_ntf, out_png, ntf_size, bbx_size, msi_ntf=None):
    (ntf_width, ntf_height) = ntf_size
    (ul_col, ul_row, width, height) = bbx_size

    # assert bounding box is completely inside the image
    assert (ul_col >= 0 and ul_col + width - 1 < ntf_width
            and ul_row >= 0 and ul_row + height - 1 < ntf_height)

    logging.info('ntf image to cut: {}, width, height: {}, {}'.format(in_ntf, ntf_width, ntf_height))
    logging.info('cut image bounding box, ul_col, ul_row, width, height: {}, {}, {}, {}'.format(ul_col, ul_row,
                                                                                                width, height))
    logging.info('png image to save: {}'.format(out_png))

    if msi_ntf is not None:
        logging.info("Found MSI: {} for PAN {}".format(msi_ntf, in_ntf))
        create_rgb_aligned_with_pan(in_ntf, ul_col, ul_row, width, height, msi_ntf, out_png)
    else:
        logging.info("No MSI match for {}".format(in_ntf))
        # note the coordinate system of .ntf
        cmd = 'gdal_translate -of png -ot UInt16 -srcwin {} {} {} {} {} {}' \
            .format(ul_col, ul_row, width, height, in_ntf, out_png)
        run_cmd(cmd)
        os.remove('{}.aux.xml'.format(out_png))
