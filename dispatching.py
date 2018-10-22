import os
import osrm
import numpy as np
import pandas as pd

from abc import abstractmethod, ABCMeta
import osrm


class BaseDispatcher(object):
    """ Base class for dispatchers. Not useful to instantiate on its own. """
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def dispatch(self, destination_coords, candidate_vehicles):
        """ Decide which vehicle to dispatch """

    def save_time_matrix(self, path="data/responsetimes/time_matrix.csv"):
        """ Save the matrix with travel durations. """
        (self.time_matrix
            .reset_index(drop=False)
            .rename(columns={0:"origin"})
            .to_csv(path, index=False))

    def load_time_matrix(self, path="data/responsetimes/time_matrix.csv"):
        """ Load a pre-calculated matrix with travel durations. """
        return pd.read_csv(path, index_col=0)

class ShortestDurationDispatcher(BaseDispatcher):
    """ Dispatcher that dispatches the vehicle with the shortest travel time
        estimated by the Open Source Routing Machine. """

    def __init__(self, demand_locs, station_locs, osrm_host="http://192.168.56.101:5000",
                 load_matrix=True, save_matrix=False, data_dir="data", verbose=True):
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
            vehicle_ids = [v.id for v in candidate_vehicles]
            vehicle_locs = [v.current_station for v in candidate_vehicles]
            options = self.time_matrix.loc[vehicle_locs, destination_loc]
            best = vehicle_ids[options.values.argmin()]
            return best, options.min()
        else:
            return "EXTERNAL", None
