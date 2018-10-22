from pyproj import Proj, transform


def lonlat_to_xy(lon, lat):
    inProj = Proj("+init=EPSG:4326")
    outProj = Proj("+init=EPSG:28992")
    x, y = transform(inProj, outProj, lon, lat)
    return x, y


def xy_to_lonlat(x, y):
    outProj = Proj("+init=EPSG:4326")
    inProj = Proj("+init=EPSG:28992")
    lon, lat = transform(inProj, outProj, x, y)
    return lon, lat


def pre_process_station_name(x):
    """
    Standarizes the station names. This step is necesary to merge
    different data sets later.
    """
    return x.upper()
