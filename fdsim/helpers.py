import numpy as np
import pandas as pd
import pickle
import time


def lonlat_to_xy(lon, lat):
    """Convert (longitude, latitude) coordinates to (x, y).

    Requires `pyproj` to be installed.
    """
    from pyproj import Proj, transform
    inProj = Proj("+init=EPSG:4326")
    outProj = Proj("+init=EPSG:28992")
    x, y = transform(inProj, outProj, lon, lat)
    return x, y


def xy_to_lonlat(x, y):
    """Convert (x, y) coordinates to (longitude, latitude).

    Requires `pyproj` to be installed.
    """
    from pyproj import Proj, transform
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


def quick_load_simulator(path="data/simulator.pickle"):
    """ Load a pickled Simulator object so that initialization can be skipped.

    Notes
    -----
    Also re-initializes the response time and incident time generator objects, since they
    cannot be pickled and were therefore deleted before saving.

    Parameters
    ----------
    path: str
        The path to the pickled Simulator object. Optional, defaults to
        'data/simulator.pickle', which is also the default of the
        Simulator.save_simulator() method.

    Returns
    -------
    The loaded Simulator object.
    """
    sim = pickle.load(open(path, "rb"))
    sim.rsampler._create_response_time_generators()
    sim.isampler.reset_time()
    sim.big_sampler._create_big_incident_generator()
    sim.set_max_target(sim.max_target)
    return sim


def progress(text, verbose=True, same_line=False, newline_end=True):
    if verbose:
        print("{}[{}] {}".format("\r" if same_line else "", time.strftime("%Y-%m-%d %H:%M:%S"),
                                 text), end="\n" if newline_end else "")


def create_service_area_dict(kvt_path, veh_col="vehicle_type", station_col="kazerne_naam",
                             loc_col="vak_nr", station_filter=None):
    """Create a dictionary of service areas based on KVT data.

    Parameters
    ----------
    kvt_path: str
        The path to the KVT data.
    veh_col, station_col, loc_col: str
        Column names that refer to the vehicle type, station name, and location id
        respectively.
    station_filter: array-like of strings, default=None
        Stations to use. If None, use all. Will be converted to upper case.

    Returns
    -------
    service_areas: dict
        A dictionary like {'vehicle type' -> {'station name' -> [loc1, loc2, ...]}}.
    """
    data = pd.read_csv(kvt_path)
    if station_filter is not None:
        data = data[np.in1d(data['kazerne_naam'], pd.Series(station_filter).str.upper())]

    grouped = (data.groupby([veh_col, station_col])[loc_col]
                   .apply(lambda x: x.tolist())
                   .reset_index(veh_col)
                   .pivot(columns=veh_col)
              )
    grouped.columns = grouped.columns.droplevel(0)

    d = grouped.to_dict()

    # remove NaNs from dictionary
    for v in data[veh_col].unique():
        for s in data[station_col].unique():
            if np.isnan(d[v][s]).all():
                del d[v][s]

    return d
