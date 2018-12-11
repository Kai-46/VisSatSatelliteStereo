# Copyright (C) 2015, Carlo de Franchis <carlo.de-franchis@cmla.ens-cachan.fr>
# Copyright (C) 2015, Gabriele Facciolo <facciolo@cmla.ens-cachan.fr>
# Copyright (C) 2015, Enric Meinhardt <enric.meinhardt@cmla.ens-cachan.fr>


import numpy as np

def apply_poly(poly, x, y, z):
    """
    Evaluates a 3-variables polynom of degree 3 on a triplet of numbers.

    Args:
        poly: list of the 20 coefficients of the 3-variate degree 3 polynom,
            ordered following the RPC convention.
        x, y, z: triplet of floats. They may be numpy arrays of same length.

    Returns:
        the value(s) of the polynom on the input point(s).
    """
    out = 0
    out += poly[0]
    out += poly[1]*y + poly[2]*x + poly[3]*z
    out += poly[4]*y*x + poly[5]*y*z +poly[6]*x*z
    out += poly[7]*y*y + poly[8]*x*x + poly[9]*z*z
    out += poly[10]*x*y*z
    out += poly[11]*y*y*y
    out += poly[12]*y*x*x + poly[13]*y*z*z + poly[14]*y*y*x
    out += poly[15]*x*x*x
    out += poly[16]*x*z*z + poly[17]*y*y*z + poly[18]*x*x*z
    out += poly[19]*z*z*z
    return out


def apply_rfm(num, den, x, y, z):
    """
    Evaluates a Rational Function Model (rfm), on a triplet of numbers.

    Args:
        num: list of the 20 coefficients of the numerator
        den: list of the 20 coefficients of the denominator
            All these coefficients are ordered following the RPC convention.
        x, y, z: triplet of floats. They may be numpy arrays of same length.

    Returns:
        the value(s) of the rfm on the input point(s).
    """
    return apply_poly(num, x, y, z) / apply_poly(den, x, y, z)


# # this function was written to use numpy.polynomial.polynomial.polyval3d
# # function, instead of our apply_poly function.
# def reshape_coefficients_vector(c):
#     """
#     Transform a 1D array of coefficients of a 3D polynom into a 3D array.
#
#     Args:
#         c: 1D array of length 20, containing the coefficients of the
#             3-variables polynom of degree 3, ordered with the RPC convention.
#     Returns:
#         a 4x4x4 ndarray, with at most 20 non-zero entries, containing the
#         coefficients of input array.
#     """
#     out = np.zeros((4, 4, 4))
#     out[0, 0, 0] = c[0]
#     out[0, 1, 0] = c[1]
#     out[1, 0, 0] = c[2]
#     out[0, 0, 1] = c[3]
#     out[1, 1, 0] = c[4]
#     out[0, 1, 1] = c[5]
#     out[1, 0, 1] = c[6]
#     out[0, 2, 0] = c[7]
#     out[2, 0, 0] = c[8]
#     out[0, 0, 2] = c[9]
#     out[1, 1, 1] = c[10]
#     out[0, 3, 0] = c[11]
#     out[2, 1, 0] = c[12]
#     out[0, 1, 2] = c[13]
#     out[1, 2, 0] = c[14]
#     out[3, 0, 0] = c[15]
#     out[1, 0, 2] = c[16]
#     out[0, 2, 1] = c[17]
#     out[2, 0, 1] = c[18]
#     out[0, 0, 3] = c[19]
#     return out
#
#
# def apply_rfm_numpy(num, den, x, y, z):
#     """
#     Alternative implementation of apply_rfm, that uses numpy to evaluate
#     polynoms.
#     """
#     c_num = reshape_coefficients_vector(num)
#     c_den = reshape_coefficients_vector(den)
#     a = np.polynomial.polynomial.polyval3d(x, y, z, c_num)
#     b = np.polynomial.polynomial.polyval3d(x, y, z, c_den)
#     return a/b


class RPCModel(object):
    def __init__(self, rpc_dict):
        # normalization constant
        self.rowOff = rpc_dict['rowOff']
        self.rowScale = rpc_dict['rowScale']

        self.colOff = rpc_dict['colOff']
        self.colScale = rpc_dict['colScale']

        self.latOff = rpc_dict['latOff']
        self.latScale = rpc_dict['latScale']

        self.lonOff = rpc_dict['lonOff']
        self.lonScale = rpc_dict['lonScale']

        self.altOff = rpc_dict['altOff']
        self.altScale = rpc_dict['altScale']

        # polynomial coefficients
        self.rowNum = rpc_dict['rowNum']
        self.rowDen = rpc_dict['rowDen']
        self.colNum = rpc_dict['colNum']
        self.colDen = rpc_dict['colDen']

    def projection(self, lat, lon, alt):
        cLon = (lon - self.lonOff) / self.lonScale
        cLat = (lat - self.latOff) / self.latScale
        cAlt = (alt - self.altOff) / self.altScale
        cCol = apply_rfm(self.colNum, self.colDen, cLat, cLon, cAlt)
        cRow = apply_rfm(self.rowNum, self.rowDen, cLat, cLon, cAlt)
        col = cCol*self.colScale + self.colOff
        row = cRow*self.rowScale + self.rowOff
        return col, row

    def inverse_projection(self, col, row, alt, return_normalized=False):
        """
        Args:
            col, row: image coordinates
            alt: altitude (in meters above the ellipsoid) of the corresponding
                3D point
            return_normalized: boolean flag. If true, then return normalized
                coordinates
        Returns:
            lon, lat, alt
        """
        # normalise input image coordinates
        cCol = (col - self.colOff) / self.colScale
        cRow = (row - self.rowOff) / self.rowScale
        cAlt = (alt - self.altOff) / self.altScale

        # target point: Xf (f for final)
        Xf = np.vstack([cCol, cRow]).T

        # use 3 corners of the lon, lat domain and project them into the image
        # to get the first estimation of (lon, lat)
        # EPS is 2 for the first iteration, then 0.1.
        lon = -np.ones(len(Xf))
        lat = -np.ones(len(Xf))
        EPS = 2
        x0 = apply_rfm(self.colNum, self.colDen, lat, lon, cAlt)
        y0 = apply_rfm(self.rowNum, self.rowDen, lat, lon, cAlt)
        x1 = apply_rfm(self.colNum, self.colDen, lat, lon + EPS, cAlt)
        y1 = apply_rfm(self.rowNum, self.rowDen, lat, lon + EPS, cAlt)
        x2 = apply_rfm(self.colNum, self.colDen, lat + EPS, lon, cAlt)
        y2 = apply_rfm(self.rowNum, self.rowDen, lat + EPS, lon, cAlt)

        # n = 0
        while not np.all((x0 - cCol) ** 2 + (y0 - cRow) ** 2 < 1e-18):
            X0 = np.vstack([x0, y0]).T
            X1 = np.vstack([x1, y1]).T
            X2 = np.vstack([x2, y2]).T
            e1 = X1 - X0
            e2 = X2 - X0
            u  = Xf - X0

            # project u on the base (e1, e2): u = a1*e1 + a2*e2
            # the exact computation is given by:
            #   M = np.vstack((e1, e2)).T
            #   a = np.dot(np.linalg.inv(M), u)
            # but I don't know how to vectorize this.
            # Assuming that e1 and e2 are orthogonal, a1 is given by
            # <u, e1> / <e1, e1>
            num = np.sum(np.multiply(u, e1), axis=1)
            den = np.sum(np.multiply(e1, e1), axis=1)
            a1 = np.divide(num, den)

            num = np.sum(np.multiply(u, e2), axis=1)
            den = np.sum(np.multiply(e2, e2), axis=1)
            a2 = np.divide(num, den)

            # use the coefficients a1, a2 to compute an approximation of the
            # point on the gound which in turn will give us the new X0
            lon += a1 * EPS
            lat += a2 * EPS

            # update X0, X1 and X2
            EPS = .1
            x0 = apply_rfm(self.colNum, self.colDen, lat, lon, cAlt)
            y0 = apply_rfm(self.rowNum, self.rowDen, lat, lon, cAlt)
            x1 = apply_rfm(self.colNum, self.colDen, lat, lon + EPS, cAlt)
            y1 = apply_rfm(self.rowNum, self.rowDen, lat, lon + EPS, cAlt)
            x2 = apply_rfm(self.colNum, self.colDen, lat + EPS, lon, cAlt)
            y2 = apply_rfm(self.rowNum, self.rowDen, lat + EPS, lon, cAlt)
            #n += 1

        #print('# of iterations: %d' % n)

        if return_normalized:
           return lon, lat, cAlt

        # else denormalize and return
        lon = lon*self.lonScale + self.lonOff
        lat = lat*self.latScale + self.latOff
        return lon, lat, alt

    def __repr__(self):
        return '''        
    ### Model ###
        rowNum = {rowNum}
        rowDen = {rowDen}   
        colNum = {colNum}
        colDen = {colDen}

    ### Scale and Offsets ###
        rowOff   = {rowOff}
        rowScale = {rowScale}
        colOff   = {colOff}
        colScale = {colScale}
        latOff   = {latOff}
        latScale = {latScale}
        lonOff   = {lonOff}
        lonScale = {lonScale}
        altOff   = {altOff}
        altScale = {altScale}'''.format(
            rowNum=self.rowNum,
            rowDen=self.rowDen,
            colNum=self.colNum,
            colDen=self.colDen,
            rowOff=self.rowOff,
            rowScale=self.rowScale,
            colOff=self.colOff,
            colScale=self.colScale,
            latOff=self.latOff,
            latScale=self.latScale,
            lonOff=self.lonOff,
            lonScale=self.lonScale,
            altOff=self.altOff,
            altScale=self.altScale)


if __name__ == '__main__':
    from lib.parse_meta import parse_meta
    meta_dict = parse_meta('/data2/kz298/dataset/core3d/performer_source_data/jacksonville/satellite_imagery/WV3/PAN/cleaned_data/17APR22163213-P1BS-501504472100_01_P004.XML')
    rpc_model = RPCModel(meta_dict['rpc'])
    print(rpc_model)
