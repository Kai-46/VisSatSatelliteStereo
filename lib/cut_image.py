import os
from lib.run_cmd import run_cmd
import logging

def cut_image(in_ntf, out_png, ntf_size, bbx_size):
    (ntf_width, ntf_height) = ntf_size
    (ul_col, ul_row, width, height) = bbx_size

    # assert bounding box is completely inside the image
    assert (ul_col >= 0 and ul_col + width - 1 < ntf_width
            and ul_row >= 0 and ul_row + height - 1 < ntf_height)

    logging.info('ntf image to cut: {}, width, height: {}, {}'.format(in_ntf, ntf_width, ntf_height))
    logging.info('cut image bounding box, ul_col, ul_row, width, height: {}, {}, {}, {}'.format(ul_col, ul_row, width, height))
    logging.info('png image to save: {}'.format(out_png))

    # note the coordinate system of .ntf
    cmd = 'gdal_translate -of png -ot UInt16 -srcwin {} {} {} {} {} {}' \
        .format(ul_col, ul_row, width, height, in_ntf, out_png)

    run_cmd(cmd)
    os.remove('{}.aux.xml'.format(out_png))
