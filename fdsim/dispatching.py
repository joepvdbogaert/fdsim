import os
import osrm
import numpy as np
import pandas as pd

from abc import abstractmethod, ABCMeta


class BaseDispatcher(object):
    """ Base class for dispatchers. Not useful to instantiate on its own.

    All classes that inherit form BasePredictor should implement 'dispatch()' to
    choose from a list of Vehicle objects, which one to dispatch to a specified
    location.
    """
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def dispatch(self, destination_coords, candidate_vehicles):
        """ Decide which vehicle to dispatch """

    def save_time_matrix(self, path="data/responsetimes/time_matrix.csv"):
        """ Save the matrix with travel durations. """
        (pd.DataFrame(self.time_matrix, columns=self.matrix_names, index=self.matrix_names)
            .reset_index(drop=False)
            .rename(columns={0: "origin"})
            .to_csv(path, index=False))

    def load_time_matrix(self, path="data/responsetimes/time_matrix.csv"):
        """ Load a pre-calculated matrix with travel durations. """
        return pd.read_csv(path, index_col=0)


class ShortestDurationDispatcher(BaseDispatcher):
    """ Dispatcher that dispatches the vehicle with the shortest travel time
        estimated by the Open Source Routing Machine (OSRM).

    Parameters
    ----------
    demand_locs: dict
        The coordinates of demand locations, as a dictionary like:
        {'demand location id' -> (longitude, latitude)}.
        Ignored when load_matrix=True.
    station_locs: dict
        The coordinates of the fire stations. Same form as demand_locs.
        Ignored when load_matrix=True.
    osrm_host: str
        The URL to the OSRM API. Ignored when load_matrix=True.
    load_matrix: boolean
        Whether to load the matrix of travel times from disk instead of
        computing it with OSRM. Defaults to True.
    save_matrix: boolean
        Whether to save the computed time matrix to disk after computing it
        with OSRM. Optional, defaults to false.
    data_dir: str
        The directory to store the time matrix and/or load it from.
    verbose: boolean
        Whether to print progress to console.
    """

    def __init__(self, demand_locs=None, station_locs=None,
                 osrm_host="http://192.168.56.101:5000", load_matrix=True,
                 save_matrix=False, data_dir="data", verbose=True):
        """ Create the matrix of travel durations with OSRM. """
        self.osrm_host = osrm_host
        self.demand_locs = demand_locs
        self.station_locs = station_locs
        self.verbose = verbose
        self.path = os.path.join(data_dir, "time_matrix.csv")

        osrm.RequestConfig.host = self.osrm_host
        self.osrm_config = osrm.RequestConfig

        if load_matrix:
            self.time_matrix = self.load_time_matrix(self.path)
        else:
            self.time_matrix = self._get_travel_durations()

        self.matrix_names = np.array(self.time_matrix.columns, dtype=str)
        self.time_matrix = np.array(self.time_matrix.values, dtype=np.float)
        self.n_original_stations = np.sum(pd.Series(self.matrix_names).str[0:2] != "13")
        self.time_matrix_stations = self.time_matrix[-self.n_original_stations:, :]
        self.station_names = self.matrix_names[-self.n_original_stations:]

        if save_matrix:
            self.save_time_matrix(self.path)

        if self.verbose: print("Dispatcher ready to go.")

    def _get_travel_durations(self):
        """ Use OSRM to find the travel durations between every set of demand
            locations and stations.
        """
        if self.verbose: print("Creating matrix of travel times...")
        coord_list = list(self.demand_locs.values()) + list(self.station_locs.values())
        id_list = list(self.demand_locs.keys()) + list(self.station_locs.keys())

        time_matrix, _, _ = osrm.table(coord_list,
                                       coords_dest=coord_list,
                                       ids_origin=id_list,
                                       ids_dest=id_list,
                                       output='dataframe',
                                       url_config=self.osrm_config)
        return time_matrix

    def set_custom_stations(self, station_locations, station_names):
        """ Set custom station locations.

        The function recreates the time_matrix_stations, but leaves the original
        time_matrix intact. The new time_matrix_stations has len(station_locations) rows
        and the same columns as time_matrix (fictive column names are still 'matrix_names').

        Parameters
        ----------
        station_locations: array-like of strings
            The demand locations that should get a fire station.
        station_names: array-like of strings
            The names of the custom stations.
        """
        assert len(station_locations) == len(station_names), ("Lengths of station_locations"
            " and station_names does not match")

        loc_indexes = [np.nonzero(self.matrix_names == loc)[0][0] for loc in station_locations]
        self.time_matrix_stations = self.time_matrix[loc_indexes, :]
        self.station_names = np.array(station_names, dtype=str)

    def dispatch(self, destination_loc, candidate_vehicles):
        """ Dispatches the vehicle with the shortest estimated response time
            according to OSRM.

        Parameters
        ----------
        destination_coords: tuple(float, float)
            The coordinates of the incident to dispatch to in decimal long lat.
        candidate_vehicles: array-like of Vehicle objects
            Vehicle objects with their current state and locations.

        Returns
        -------
        ID of the vehicle to dispatch.
        """
        if len(candidate_vehicles) > 0:
            # save IDs and locations as lists
            vehicle_ids = np.array([v.id for v in candidate_vehicles], dtype=str)
            vehicle_locs = np.array([v.current_station for v in candidate_vehicles], dtype=str)
            # create subset of time_matrix corresponding to available vehicles
            mask = np.in1d(self.station_names, vehicle_locs)
            dest_idx = np.nonzero(self.matrix_names == destination_loc)[0][0]
            options = self.time_matrix_stations[mask, dest_idx]
            # choose closest station and corresponding vehicle ID
            best_station = self.station_names[mask][options.argmin()]
            best_vehicle_id = vehicle_ids[np.nonzero(vehicle_locs == best_station)][0]
            return best_vehicle_id, options.min()
        else:
            return "EXTERNAL", None
