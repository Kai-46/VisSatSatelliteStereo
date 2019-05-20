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
