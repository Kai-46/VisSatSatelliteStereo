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

import pymap3d


def latlonalt_to_enu(lat, lon, alt, lat0, lon0, alt0):
    e, n, u = pymap3d.geodetic2enu(lat, lon, alt, lat0, lon0, alt0)

    return e, n, u


def enu_to_latlonalt(e, n, u, lat0, lon0, alt0):
    lat, lon, alt = pymap3d.enu2geodetic(e, n, u, lat0, lon0, alt0)

    return lat, lon, alt


if __name__ == '__main__':
    lat = -34.450
    lon = -58.579
    alt = 20.31

    lat0 = -34.448
    lon0 = -58.577
    alt0 = -30.0

    e, n, u = latlonalt_to_enu(lat, lon, alt, lat0, lon0, alt0)
    lat, lon, alt = enu_to_latlonalt(e, n, u, lat0, lon0, alt0)

    print('{}, {}, {}'.format(e, n, u))
    print('{}, {}, {}'.format(lat, lon, alt))
