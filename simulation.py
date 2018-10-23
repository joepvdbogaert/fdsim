import os
import numpy as np
import pandas as pd

from sampling import IncidentSampler, ResponseTimeSampler
from objects import Vehicle
from dispatching import ShortestDurationDispatcher

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
    load_response_data: boolean
        Whether to load preprocessed response data from disk (True) or to
        calculate it using OSRM.
    load_time_matrix: boolean
        Whether to load the matrix of travel durations from disk (True) or
        to calculate it using OSRM.
    save_response_data: boolean
        Whether to save the prepared response data with OSRM estimates to disk.
    save_time_matrix: boolean
        Whether to save the matrix of travel durations to disk.
    vehicle_types: array-like of strings
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
        The name of the column that identifies the demand locations, defaults to
        'hub_vak_bk'. This is also the only currently supported value.
    verbose: boolean
        Whether to print progress updates to the console during computations.

    Example
    -------
    >>> from simulation import Simulator
    >>> sim = Simulator(incidents, deployments, stations, vehicle_allocation)
    >>> sim.simulate_n_incidents(10000)
    >>> sim.save_log("simulation_results.csv")

    Continue simulating where you left of:
    >>> sim.simulate_n_incidents(10000, restart=False)

    You can save the simulor object after initializing, so that next time you can
    skip the initialization (requires the _pickle_ module):
    >>> sim.save_simulator_state()
    >>> sim = pickle.load(open('simulator.pickle', 'rb'))
    """
    def __init__(self, incidents, deployments, stations, vehicle_allocation,
                 load_response_data=True, load_time_matrix=True, save_response_data=False,
                 save_time_matrix=False, vehicle_types=["TS", "RV", "HV", "WO"],
                 predictor="prophet", start_time=None, end_time=None, data_dir="data",
                 osrm_host="http://192.168.56.101:5000", location_col="hub_vak_bk",
                 verbose=True):

        self.data_dir = os.path.join(ROOT_DIR, data_dir)
        self.verbose = verbose

        self.rsampler = ResponseTimeSampler(load_data=load_response_data,
                                            data_dir=self.data_dir,
                                            verbose=verbose)

        self.rsampler.fit(incidents=incidents, deployments=deployments, stations=stations,
                          vehicle_types=vehicle_types, osrm_host=osrm_host,
                          save_prepared_data=save_response_data, location_col=location_col)

        locations = list(self.rsampler.location_coords.keys())
        self.isampler = IncidentSampler(incidents, deployments, vehicle_types, locations,
                                        start_time=start_time, end_time=end_time,
                                        predictor=predictor, verbose=verbose)

        self.vehicles = self._create_vehicle_dict(vehicle_allocation)

        self.dispatcher = ShortestDurationDispatcher(demand_locs=self.rsampler.location_coords,
                                                     station_locs=self.rsampler.station_coords,
                                                     osrm_host=osrm_host,
                                                     load_matrix=load_time_matrix,
                                                     save_matrix=save_time_matrix,
                                                     data_dir=self.data_dir,
                                                     verbose=verbose)

        if start_time is not None:
            self.start_time = start_time
        else:
            self.start_time = self.isampler.sampling_dict[0]["time"]
        self.end_time = end_time

        if self.verbose: print("Simulator is ready. At your service.")

    def _create_vehicle_dict(self, vehicle_allocation):
        """ Create a dictionary of Vehicle objects from the station data.

        Parameters
        ----------
        vehicle_allocation: pd.DataFrame
            The allocation of vehicles to stations. Column names should represent
            vehicle types, row names station names, and values are the number of
            vehicles assigned to the station.

        Returns
        -------
        A dictionary like: {'vehicle id' -> Vehicle object}.
        """
        vehicle_allocation["kazerne"] = vehicle_allocation["kazerne"].str.upper()
        vehicle_allocation.rename(columns={"#WO": "WO", "#TS": "TS", "#RV": "RV", "#HV": "HV"},
                                  inplace=True)
        vs = vehicle_allocation.set_index("kazerne").unstack().reset_index()
        vdict = {}
        id_counter = 1
        for r in range(len(vs)):
            for _ in range(vs[0].iloc[r]):
                coords = self._get_station_coordinates(vs["kazerne"].iloc[r])
                this_id = "V" + str(id_counter)
                vdict[this_id] = Vehicle(this_id, vs["level_0"].iloc[r],
                                         vs["kazerne"].iloc[r], coords)
                id_counter += 1

        return vdict

    def _get_coordinates(self, demand_location):
        """ Get the coordinates of a demand location.

        Parameters
        ----------
        demand_location: str
            The ID of the demand location to get the coordinates of.
        """
        return self.rsampler.locations_coords[demand_location]

    def _get_station_coordinates(self, station_name):
        """ Get the coordinates of a fire station.

        Parameters
        ----------
        station_name: str
            The name of station to get the coordinates of.
        """
        return self.rsampler.station_coords[station_name]

    def _sample_incident(self, t):
        """ Sample the next incident.

        Parameters
        ----------
        t: float
            The time in minutes since the simulation start. From t, the exact timestamp is
            determined, which leads to certain incident rates for different incident types.
        """
        t, time, type_, loc, prio, req_vehicles, func = self.isampler.sample_next_incident(t)
        destination_coords = self.rsampler.location_coords[loc]
        return t, time, type_, loc, prio, req_vehicles, func, destination_coords

    def _pick_vehicle(self, location, vehicle_type):
        """ Dispatch a vehicle to coordinates.

        Parameters
        ----------
        location: str
            The ID of the demand location where a vehicle should be dispatched to.
        vehicle_type: str
            The type of vehicle to send to the incident.

        Returns
        -------
        The Vehicle object of the chosen vehicle and the estimated travel time to
        the incident / demand location in seconds.
        """
        candidates = [self.vehicles[v] for v in self.vehicles.keys() if
                      self.vehicles[v].available and self.vehicles[v].type == vehicle_type]

        vehicle_id, estimated_time = self.dispatcher.dispatch(location, candidates)

        if vehicle_id == "EXTERNAL":
            return None, None

        vehicle = self.vehicles[vehicle_id]

        return vehicle, estimated_time

    def _update_vehicles(self, t):
        """ Return vehicles that finished their jobs to their base stations.

        Parameters
        ----------
        t: float
            The time since the start of the simulation. Determines which vehicles
            have become available again and can return to their bases.
        """
        for vehicle in self.vehicles.values():
            if not vehicle.available and vehicle.becomes_available < t:
                vehicle.return_to_base()

    def _prepare_results(self):
        """ Create pd.DataFrame with descriptive column names of logged results."""
        self.results = pd.DataFrame(self.log[0:self.log_index, :], columns=self.log_columns)

    def simulate_n_incidents(self, N, restart=True):
        """ Simulate N incidents and their reponses.

        Parameters
        ----------
        N: int
            The number of incidents to simulate.
        restart: boolean
            Whether to empty the log and reset time before simulation (True)
            or to continue where stopped (False). Optional, defaults to True.
        """
        if restart:
            self._initialize_log(N)
            t = 0
        else:
            t = self.log[0:self.log_index, 0].max()

        for _ in range(N):
            # sample incident and update status of vehicles at new time t
            t, time, type_, loc, prio, req_vehicles, func, dest = self._sample_incident(t)
            self._update_vehicles(t)

            for v in req_vehicles:

                vehicle, estimated_time = self._pick_vehicle(loc, v)
                if vehicle is None:
                    dispatch, turnout, travel, onscene, response = [np.nan]*5
                    self._log([t, time, type_, loc, prio, func, v, "EXTERNAL", dispatch,
                               turnout, travel, onscene, response, "EXTERNAL", "EXTERNAL"])
                else:
                    dispatch, turnout, travel, onscene, response = \
                        self.rsampler.sample_response_time(
                            type_, loc, vehicle.current_station,
                            vehicle.type, estimated_time=estimated_time)

                    vehicle.dispatch(dest, t + (onscene/60) + (estimated_time/60))

                    self._log([t, time, type_, loc, prio, func, vehicle.type, vehicle.id,
                               dispatch, turnout, travel, onscene, response,
                               vehicle.current_station, vehicle.base_station])

        self._prepare_results()
        print("Simulated {} incidents. See Simulator.results for the results.".format(N))

    def _initialize_log(self, N):
        """ Create an empty log.

        Initializes an empty self.log and self.log_index = 0. Nothing is returned. Log is
        initialized of certain size to avoid slow append / concatenate operations.
        Creates an 3*N size log because incidents can have multiple log entries,
        unless 3*N > one million, then initialize of size one-million and extend
        on the fly when needed to avoid ennecessary memory issues.

        Parameters
        ----------
        N: int
            The number of incidents that will be simulated.
        """
        if 3*N > 1000000:
            # initialize smaller and add more rows later.
            size = 1000000
        else:
            # multiple deployments per incident
            size = 3*N

        self.log = np.empty((size, 15), dtype=object)
        self.log_columns = ["t", "time", "incident_type", "location", "priority",
                            "object_function", "vehicle_type", "vehicle_id", "dispatch_time",
                            "turnout_time", "travel_time", "on_scene_time", "response_time",
                            "station", "base_station_of_vehicle"]
        self.log_index = 0

    def _log(self, values):
        """ Insert values in the log.

        When log is 'full', the log size is doubled by concatenating an empty array of the
        same shape. This ensures that only few concatenate operations are required (this is
        desirable because they are relatively slow).

        Parameters
        ----------
        values: array-like
            Must contain the values in the specific order as specified in 'self.log_columns'.
        """
        try:
            self.log[self.log_index, :] = values
        except IndexError:
            # if ran out of array size, add rows and continue
            print("Log full at: {} entries, log extended.".format(self.log_index))
            self.log = np.concatenate([self.log,
                                       np.empty(self.log.shape, dtype=object)],
                                      axis=0)
            self.log[self.log_index, :] = values

        self.log_index += 1

    def save_log(self, file_name="simulation_results.csv"):
        """ Save the current log file to disk.

        Notes
        -----
        File gets saved in the folder 'data_dir' that is specified on
        initialization of the Simulator.

        Parameters
        ----------
        file_name: str
            How to name the csv of the log.
        """
        self.results.to_csv(os.path.join(self.data_dir, file_name), index=False)

    def save_simulator_object(self):
        """ Save the Simulator instance as a pickle for quick loading.

        Saves the entire Simulator object as a pickle, so that it can be quickly loaded
        with all preprocessed attributes. Note: generator objects are not supported by
        pickle, so they have to be removed before dumping and re-initialized after
        loading.

        Notes
        -----
        Requires the pickle package to be installed.
        """
        try:
            import pickle
            del self.rsampler.dispatch_generators
            del self.rsampler.turnout_generators
            del self.rsampler.travel_time_noise_generators
            del self.rsampler.onscene_generators
            pickle.dump(self, open(os.path.join(self.data_dir, "simulator.pickle"), "wb"))
        except ImportError:
            print("This method requires the pickle package, which is not installed. Install"
                  " with 'pip install pickle' or 'conda install pickle'.")
