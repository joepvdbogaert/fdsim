import os
import numpy as np
import pandas as pd

from sampling import IncidentSampler, ResponseTimeSampler
from objects import Vehicle, DemandLocation, IncidentType
from dispatching import ShortestDurationDispatcher
from helpers import pre_process_station_name

from definitions import ROOT_DIR


class Simulator():
    """ Main simulator class that simulates incidents and reponses at the fire department.

    Parameters
    ----------
    incidents: pd.DataFrame
        The incident data.
    deployments: pd.DataFrame (optional)
        The deployment data.
    stations: pd.DataFrame (optional)
        The station information including coordinates and station names.
    vehicle_allocation: pd.DataFrame
        The allocation of vehicles to stations. Expected columns:
        ["kazerne", "#TS", "#RV", "#HV", "#WO"].
    vehicles: array-like of strings
        The vehicle types to incorporate in the simulation. Optional, defaults
        to ["TS", "RV", "HV", "WO"].
    predictor: str
        Type of predictor to use. Defaults to 'prophet', which uses Facebook's
        Prophet package to forecast incident rate per incident type based on trend
        and yearly, weekly, and daily patterns.
    start_time: Timestamp or str (convertible to timestamp)
        The start of the time period to simulate. If None, forecasts from
        the end of the data. Defaults to None.
    end_time: Timestamp or str (convertible to timestamp)
        The end of the time period to simulate. If None, forecasts until one year
        after the end of the data. Defaults to None.
    data_dir: str
        The path to the directory where data should be loaded from and saved to.
        Defaults to '/data/'.
    osrm_host: str
        URL to the OSRM API, defaults to 'http://192.168.56.101:5000'
    location_col: str
        The name of the column that identifies the demand locations, defaults to 'hub_vak_bk'.
    verbose: boolean
        Whether to print progress updates to the console during computations.

    Example
    -------

    """
    def __init__(self, incidents, deployments, stations, vehicle_allocation,
                 load_response_data=True, vehicles=["TS", "RV", "HV", "WO"],
                 predictor="prophet", start_time=None, end_time=None, data_dir="data",
                 osrm_host="http://192.168.56.101:5000", location_col="hub_vak_bk",
                 verbose=True):

        self.data_dir = os.path.join(ROOT_DIR, data_dir)
        self.rsampler = ResponseTimeSampler(load_data=True, data_dir=self.data_dir,
                                            verbose=verbose)
        self.rsampler.fit()
        self.isampler = IncidentSampler(incidents, deployments, vehicles,
                                        start_time=start_time, end_time=end_time,
                                        predictor=predictor, verbose=verbose)

        self.vehicles = self._create_vehicle_dict(vehicle_allocation)
        self.dispatcher = ShortestDurationDispatcher(osrm_host=osrm_host)

        if start_time is not None:
            self.start_time = start_time
        else:
            self.start_time = self.isampler.sampling_dict[0]["time"]
        self.end_time = end_time

    def _create_vehicle_dict(self, vehicle_allocation):
        """ Create a dictionary of Vehicle objects from the station data. """
        vehicle_allocation["kazerne"] = vehicle_allocation["kazerne"].str.upper()
        vehicle_allocation.rename(columns={"#WO": "WO", "#TS": "TS", "#RV": "RV", "#HV": "HV"},
                                  inplace=True)
        vs = vehicle_allocation.set_index("kazerne").unstack().reset_index()
        print(self.rsampler.station_coords.keys())
        print(vs["kazerne"].unique())
        print(len(vs["kazerne"].unique()))
        vdict = {}
        id_counter = 1
        for r in range(len(vs)):
            for _ in range(vs[0].iloc[r]):
                coords = self._get_station_coordinates(vs["kazerne"].iloc[r])
                vdict["V" + str(id_counter)] = Vehicle(id_counter, vs["level_0"].iloc[r],
                                                       vs["kazerne"].iloc[r], coords)
                id_counter += 1

        return vdict

    def _get_coordinates(self, demand_location):
        """ Get the coordinates of a demand location """
        return self.rsampler.locations_coords[demand_location]

    def _get_station_coordinates(self, station_name):
        """ Get the coordinates of a statio. """
        return self.rsampler.station_coords[station_name]

    def _initialize_log(self, N):
        """ Create an empty log. """
        self.log = pd.DataFrame(np.zeros((N, 15)),
            columns=["t", "time", "incident_type", "location", "priority", "object_function",
                     "vehicle_type", "vehicle_id", "dispatch_time", "turnout_time",
                     "travel_time", "on_scene_time", "response_time", "station",
                     "base_station_of_vehicle"])
        self.log_index = 0

    def _log(self, values):
        """ Insert values in the log. """
        self.log.iloc[self.log_index, :].values = values

    def save_log(self, file_name="simulation_results.csv"):
        """ Save the current log file to disk. """
        self.log.to_csv(os.path.join(self.data_dir, file_name))

    def _sample_incident(self, t):
        """ Sample the next incident """
        t, time, type_, loc, prio, req_vehicles, func = self.isampler.sample_next_incident(t)
        destination_coords = self.rsampler.location_coords[loc]
        return t, time, type_, loc, prio, req_vehicles, func, destination_coords

    def _pick_vehicles(self, coordinates, vehicle_type):
        """ Dispatch a vehicle to coordinates. """
        options = [self.vehicles[v] for v in self.vehicles.keys() if 
                   self.vehicles[v].available and self.vehicles[v].type == vehicle_type]

        vehicle_id = self.dispatcher.dispatch(coordinates, options)
        vehicle = self.vehicles[vehicle_id]

        return vehicle

    def _update_vehicles(self, t):
        """ Return vehicles that finished a job to their bases. """
        for vehicle in self.vehicles.values():
            if not vehicle.is_available() and vehicle.becomes_available < t:
                vehicle.return_to_base()

    def save_simulator_state(self):
        import pickle
        pickle.dump(self, os.path.join(self.data_dir, "simulator.pickle"))

    def simulate_n_incidents(self, N):
        """ Simulate N incidents and their reponses.

        Parameters
        ----------
        N: int
            The number of incidents to simulate.

        Returns
        -------
        Float, the proportion of incidents that was served on time.
        """

        self._initialize_log(N)
        t = 0
        for _ in range(N):
            # sample incident and update status of vehicles at new time t
            t, time, type_, loc, prio, req_vehicles, func, dest = self._sample_incident(t)
            self._update_vehicles(t)

            for v in req_vehicles:

                vehicle = self._pick_vehicles(dest, v)

                dispatch, turnout, travel, onscene, response = \
                    self.rsampler.sample_response_time(type_, loc, vehicle.current_station,
                                                       vehicle.type)
                vehicle.dispatch(dest, t + (onscene/60))

                self._log([t, time, type_, loc, prio, func, vehicle.type, vehicle.id,
                           dispatch, turnout, travel, onscene, response,
                           vehicle.current_station, vehicle.base_station])
