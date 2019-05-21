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
