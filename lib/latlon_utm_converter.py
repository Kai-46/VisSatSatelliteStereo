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


import utm
import numpy as np
import pyproj


# pyproj implementation of the coordinate conversion
def latlon_to_eastnorh(lat, lon):
    # assume all the points are either on north or south hemisphere
    assert(np.all(lat >= 0) or np.all(lat < 0))

    if lat[0, 0] >= 0: # north hemisphere
        south = False
    else:
        south = True

    _, _, zone_number, _ = utm.from_latlon(lat[0, 0], lon[0, 0])

    proj = pyproj.Proj(proj='utm', ellps='WGS84', zone=zone_number, south=south)
    east, north = proj(lon, lat)
    return east, north


def eastnorth_to_latlon(east, north, zone_number, hemisphere):
    if hemisphere == 'N':
        south = False
    else:
        south = True

    proj = pyproj.Proj(proj='utm', ellps='WGS84', zone=zone_number, south=south)
    lon, lat = proj(east, north, inverse=True)
    return lat, lon


if __name__ == '__main__':
    # a point on north hemisphere
    lat = np.array([47.9941214]).reshape((1, 1))
    lon = np.array([7.8509671]).reshape((1, 1))
    print('lat, lon: {}'.format(np.hstack((lat, lon))))

    east, north = latlon_to_eastnorh(lat, lon)
    print('previous utm: {}'.format(np.hstack((east, north))))

    east, north = latlon_to_eastnorh(lat, lon)
    print('current utm: {}'.format(np.hstack((east, north))))

    lat, lon = eastnorth_to_latlon(east, north, zone_number=32, hemisphere='N')
    print('lat, lon: {}'.format(np.hstack((lat, lon))))

    print('\n')
    lat = np.array([-47.9941214]).reshape((1, 1))
    lon = np.array([7.8509671]).reshape((1, 1))
    print('lat, lon: {}'.format(np.hstack((lat, lon))))

    east, north = latlon_to_eastnorh(lat, lon)
    print('previous utm: {}'.format(np.hstack((east, north))))

    east, north = latlon_to_eastnorh(lat, lon)
    print('current utm: {}'.format(np.hstack((east, north))))

    lat, lon = eastnorth_to_latlon(east, north, zone_number=32, hemisphere='S')
    print('lat, lon: {}'.format(np.hstack((lat, lon))))
