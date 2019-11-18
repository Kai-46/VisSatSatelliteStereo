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


def get_aoi_dict(kml_file):
    with open(kml_file) as fp:
        content = fp.read()
    idx1 = content.find('<coordinates>')
    idx2 = content.find('</coordinates>')

    vertices = []
    tmp = content[idx1+len('<coordinates>'):idx2]
    tmp = tmp.split('\n')
    for line in tmp:
        line = line.strip()
        if line:
            vertices.append([float(x) for x in line.split(',')])

    # convert to UTM
    vertices_utm = [utm.from_latlon(vertex[1], vertex[0]) for vertex in vertices]

    easts = [vertex[0] for vertex in vertices_utm]
    norths = [vertex[1] for vertex in vertices_utm]

    ul_east = min(easts)
    ul_north = max(norths)
    w = max(easts) - ul_east
    h = ul_north - min(norths)

    zone_number = vertices_utm[0][2]
    zone_letter = vertices_utm[0][3]

    aoi_dict = {
        'zone_number': zone_number,
        'zone_letter': zone_letter,
        'x': ul_east,
        'y': ul_north,
        'w': w,
        'h': h
    }
    return aoi_dict

if __name__ == '__main__':
    kml_file = '/data2/kz298/dataset/mvs3dm/Challenge_Data_and_Software/kml_polygons/Explorer.kml'
    aoi_dict = get_aoi_dict(kml_file)
    print(aoi_dict)

