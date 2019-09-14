# ===============================================================================================================
#  This file is part of Creation of Operationally Realistic 3D Environment (CORE3D).                            =
#  Copyright 2019 Cornell University - All Rights Reserved                                                      =
#  -                                                                                                            =
#  NOTICE: All information contained herein is, and remains the property of General Electric Company            =
#  and its suppliers, if any. The intellectual and technical concepts contained herein are proprietary          =
#  to General Electric Company and its suppliers and may be covered by U.S. and Foreign Patents, patents        =
#  in process, and are protected by trade secret or copyright law. Dissemination of this information or         =
#  reproduction of this material is strictly forbidden unless prior written permission is obtained              =
#  from General Electric Company.                                                                               =
#  -                                                                                                            =
#  The research is based upon work supported by the Office of the Director of National Intelligence (ODNI),     =
#  Intelligence Advanced Research Projects Activity (IARPA), via DOI/IBC Contract Number D17PC00287.            =
#  The U.S. Government is authorized to reproduce and distribute copies of this work for Governmental purposes. =
# ===============================================================================================================

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

