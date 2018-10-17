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


class ShortestDurationDispatcher(BaseDispatcher):
    """ Dispatcher that dispatches the vehicle with the shortest travel time
        estimated by the Open Source Routing Machine. """

    def __init__(self, osrm_host="http://192.168.56.101:5000"):
        self.osrm_host = osrm_host

    def dispatch(self, destination_coords, candidate_vehicles):
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
        coord_list = [list(v.coords) for v in candidate_vehicles]
        coord_list.append(list(destination_coords))

        id_list = [v.id for v in candidate_vehicles]
        id_list.append("destination")

        osrm.RequestConfig.host = self.osrm_host
        time_matrix, _, _ = osrm.table(coord_list, ids_origin=id_list,
                                       output='dataframe')
        print(time_matrix["destination"])
        print(time_matrix["destination"].idxmin())
        return time_matrix["destination"].idxmin()
