import numpy as np
from osgeo import gdal, gdal_array
import os


def parse_proj_str(proj_str):
    idx1 = proj_str.find('UTM zone')
    idx2 = proj_str.find('",')

    sub_str = proj_str[idx1:idx2]
    hemisphere = sub_str[-1]
    zone_number = int(sub_str[-3:-1])

    return zone_number, hemisphere


def read_dsm_tif(file):
    ds = gdal.Open(file)
    geo = ds.GetGeoTransform()
    proj = ds.GetProjection()
    meta = ds.GetMetadata()
    width = ds.RasterXSize
    height = ds.RasterYSize

    assert (ds.RasterCount == 1)     # dsm is only one band
    band = ds.GetRasterBand(1)       # band index is one-based

    data_type = gdal_array.GDALTypeCodeToNumericTypeCode(band.DataType)
    assert (isinstance(data_type(), np.float32))
    image = np.zeros((ds.RasterYSize, ds.RasterXSize), dtype=np.float32)

    image[:, :] = band.ReadAsArray()
    nodata = band.GetNoDataValue()

    # to ease later processing, replace nodata regions with nan
    if nodata is not None:
        mask = np.isclose(image, nodata)
        image[mask] = np.nan

    zone_number, hemisphere = parse_proj_str(proj)
    # return a meta dict
    meta_dict = {
        'geo': geo,
        'proj': proj,
        'meta': meta,
        'img_width': width,
        'img_height': height,
        'nodata': nodata,
        'zone_number': zone_number,
        'hemisphere': hemisphere,
        'ul_easting': geo[0],
        'ul_northing': geo[3],
        'east_resolution': geo[1],
        'north_resolution': abs(geo[5])
    }

    # add other info for convenience
    # lr_easting are left-margin of the last column
    # lr_northing are top-margin of the last row
    meta_dict['lr_easting'] = meta_dict['ul_easting'] + (meta_dict['img_width']-1) * meta_dict['east_resolution']
    meta_dict['lr_northing'] = meta_dict['ul_northing'] - (meta_dict['img_height']-1) * meta_dict['north_resolution']
    meta_dict['area_width'] = meta_dict['lr_easting'] - meta_dict['ul_easting']
    meta_dict['area_height'] = meta_dict['ul_northing'] - meta_dict['lr_northing']
    meta_dict['alt_min'] = float(np.nanmin(image))  # for json serialization
    meta_dict['alt_max'] = float(np.nanmax(image))

    del ds
    return image, meta_dict


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


def write_dsm_tif(image, meta_dict, out_file):
    image = image.copy()    # avoid modify source data
    driver = get_driver(out_file)

    # replace nan with no_data
    image[np.isnan(image)] = meta_dict['nodata']

    assert (len(image.shape) == 2)      # image should only be 2D

    out = driver.Create(out_file, image.shape[1], image.shape[0], 1,
                        gdal_array.NumericTypeCodeToGDALTypeCode(np.float32))

    band = out.GetRasterBand(1)     # one-based index
    band.WriteArray(image.astype(np.float32), 0, 0)
    band.SetNoDataValue(meta_dict['nodata'])
    band.FlushCache()

    out.SetGeoTransform(meta_dict['geo'])
    out.SetProjection(meta_dict['proj'])
    out.SetMetadata(meta_dict['meta'])

    del out


def modify_dsm_tif_nodata(in_file, out_file, nodata):
    image, meta_dict = read_dsm_tif(in_file)
    meta_dict['nodata'] = nodata
    write_dsm_tif(image, meta_dict, out_file)


if __name__ == "__main__":
    gt_tif = '/data2/kz298/mvs3dm_result/MasterSequesteredPark/evaluation/eval_ground_truth.tif'
    image, meta_dict = read_dsm_tif(gt_tif)

    import json
    with open('/data2/kz298/tmp.json', 'w') as fp:
        json.dump(meta_dict, fp, indent=2)
    write_dsm_tif(image, meta_dict, '/data2/kz298/tmp.tif')
    print('hello')
