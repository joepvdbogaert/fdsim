import os
import numpy as np
import pandas as pd
import pickle

from fdsim.sampling import IncidentSampler, ResponseTimeSampler
from fdsim.objects import Vehicle
from fdsim.dispatching import ShortestDurationDispatcher


class Simulator():
    """ Main simulator class that simulates incidents and reponses at the fire department.

    Parameters
    ----------
    incidents: pd.DataFrame
        The incident data.
    deployments: pd.DataFrame
        The deployment data.
    stations: pd.DataFrame
        The station information including coordinates and station names.
    vehicle_allocation: pd.DataFrame
        The allocation of vehicles to stations. Expected columns:
        ["kazerne", "#TS", "#RV", "#HV", "#WO"].
    load_response_data: boolean, optional
        Whether to load preprocessed response data from disk (True) or to
        calculate it using OSRM.
    load_time_matrix: boolean, optional
        Whether to load the matrix of travel durations from disk (True) or
        to calculate it using OSRM.
    save_response_data: boolean, optional
        Whether to save the prepared response data with OSRM estimates to disk.
    save_time_matrix: boolean, optional
        Whether to save the matrix of travel durations to disk.
    vehicle_types: array-like of strings, optional
        The vehicle types to incorporate in the simulation. Optional, defaults
        to ["TS", "RV", "HV", "WO"].
    predictor: str, optional
        Type of predictor to use. Defaults to 'prophet', which uses Facebook's
        Prophet package to forecast incident rate per incident type based on trend
        and yearly, weekly, and daily patterns.
    start_time: Timestamp or str (convertible to timestamp), optional
        The start of the time period to simulate. If None, forecasts from
        the end of the data. Defaults to None.
    end_time: Timestamp or str (convertible to timestamp), optional
        The end of the time period to simulate. If None, forecasts until one year
        after the end of the data. Defaults to None.
    data_dir: str, optional
        The path to the directory where data should be loaded from and saved to.
        Defaults to '/data/'.
    osrm_host: str, optional
        URL to the OSRM API, defaults to 'http://192.168.56.101:5000'
    location_col: str, optional
        The name of the column that identifies the demand locations, defaults to
        'hub_vak_bk'. This is also the only currently supported value.
    verbose: boolean, optional
        Whether to print progress updates to the console during computations.

    Examples
    --------
    >>> from simulation import Simulator
    >>> sim = Simulator(incidents, deployments, stations, vehicle_allocation)
    >>> sim.simulate_n_incidents(10000)
    >>> sim.save_log("simulation_results.csv")

    Continue simulating where you left of:
    ```
    >>> sim.simulate_n_incidents(10000, restart=False)
    ```
    You can save the simulor object after initializing, so that next time you can
    skip the initialization (requires the _pickle_ module):
    ```
    >>> sim.save_simulator_object()
    >>> sim = pickle.load(open('simulator.pickle', 'rb'))
    ```
    """
    # the target response times
    target_dict = {'Bijeenkomstfunctie' : 10,
                   'Industriefunctie' : 8,
                   'Woonfunctie' : 8,
                   'Straat' : 10,
                   'Overige gebruiksfunctie' : 10,
                   'Kantoorfunctie' : 10,
                   'Logiesfunctie' : 8,
                   'Onderwijsfunctie' : 8,
                   'Grachtengordel' : 10,
                   'Overig' : 10,
                   'Winkelfunctie' : 5,
                   'Kanalen en rivieren' : 10,
                   'nan' : 10,
                   'Trein' : 5,
                   'Sportfunctie' : 10,
                   'Regionale weg' : 10,
                   'Celfunctie' : 5,
                   'Tram' : 5,
                   'Metro': 5,
                   'Sloten en Vaarten' : 10,
                   'Gezondheidszorgfunctie' : 8,
                   'Lokale weg' : 5,
                   'Polders' : 10,
                   'Haven' : 10,
                   'Autosnelweg' : 10,
                   'Meren en plassen' : 10,
                   'Hoofdweg' : 10,
                   'unknown': 10}

    def __init__(self, incidents, deployments, stations, vehicle_allocation,
                 load_response_data=True, load_time_matrix=True, save_response_data=False,
                 save_time_matrix=False, vehicle_types=["TS", "RV", "HV", "WO"],
                 predictor="prophet", start_time=None, end_time=None, data_dir="data",
                 osrm_host="http://192.168.56.101:5000", location_col="hub_vak_bk",
                 verbose=True):

        self.data_dir = data_dir
        self.verbose = verbose
        self.stations = stations
        self.vehicle_allocation = self._preprocess_vehicle_allocation(vehicle_allocation)
        self.original_vehicle_allocation = self.vehicle_allocation.copy()

        self.rsampler = ResponseTimeSampler(load_data=load_response_data,
                                            data_dir=self.data_dir,
                                            verbose=verbose)

        self.rsampler.fit(incidents=incidents, deployments=deployments, stations=stations,
                          vehicle_types=vehicle_types, osrm_host=osrm_host,
                          save_prepared_data=save_response_data, location_col=location_col)

        locations = list(self.rsampler.location_coords.keys())
        self.isampler = IncidentSampler(incidents, deployments, vehicle_types, locations,
                                        start_time=start_time, end_time=end_time,
                                        predictor=predictor,
                                        fc_dir=os.path.join(data_dir),
                                        verbose=verbose)

        self.vehicles = self._create_vehicle_dict(vehicle_allocation)
        self.reset_closing_times()

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
        if end_time is not None:
            self.end_time = end_time
        else:
            self.end_time = self.isampler.sampling_dict[
                np.max(list(self.isampler.sampling_dict.keys()))]["time"]

        if self.verbose: print("Simulator is ready. At your service.")

    @staticmethod
    def _preprocess_vehicle_allocation(vehicle_allocation):
        """ Pre-process the vehicle allocation dataframe to ensure consistency. """

        # if 'kazerne' is not a column, use index as station names
        if "kazerne" not in vehicle_allocation.columns:
            station_col = vehicle_allocation.index.name
            vehicle_allocation.reset_index(inplace=True, drop=False)
            vehicle_allocation.rename(columns={station_col: "kazerne"}, inplace=True)

        vehicle_allocation["kazerne"] = vehicle_allocation["kazerne"].str.upper()
        vehicle_allocation.rename(columns={"#WO": "WO", "#TS": "TS", "#RV": "RV", "#HV": "HV"},
                                  inplace=True)
        return vehicle_allocation

    def _create_vehicle_dict(self, vehicle_allocation):
        """ Create a dictionary of Vehicle objects from the station data.

        Parameters
        ----------
        vehicle_allocation: pd.DataFrame
            The allocation of vehicles to stations. Column names should represent
            vehicle types, except for one column named 'kazerne', which specifies the
            station names. Values are the number of vehicles assigned to the station.

        Returns
        -------
        A dictionary like: {'vehicle id' -> Vehicle object}.
        """
        vehicle_allocation = self._preprocess_vehicle_allocation(vehicle_allocation)
        vs = vehicle_allocation.set_index("kazerne").unstack().reset_index()
        vdict = {}
        id_counter = 1
        for r in range(len(vs)):
            for _ in range(vs[0].iloc[r]):
                coords = self._get_station_coordinates(vs["kazerne"].iloc[r])
                this_id = "VEHICLE " + str(id_counter)
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
        return self.rsampler.location_coords[demand_location]

    def _get_station_coordinates(self, station_name):
        """ Get the coordinates of a fire station.

        Parameters
        ----------
        station_name: str
            The name of station to get the coordinates of.
        """
        return self.rsampler.station_coords[station_name]

    def _sample_incident(self):
        """ Sample the next incident.

        Parameters
        ----------
        t: float
            The time in minutes since the simulation start. From t, the exact timestamp is
            determined, which leads to certain incident rates for different incident types.
        """
        t, time, type_, loc, prio, req_vehicles, func = self.isampler.sample_next_incident()
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

    def _update_vehicles(self, t, time):
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

            station = vehicle.current_station
            hour_of_day = time.hour
            scd = self.station_closed_dict[station]

            if scd[hour_of_day]:
                # station is closed, force a dispatch to make it unavailable
                minutes = t % 60
                minutes_till_open = (scd["open_from"] > hour_of_day) * (scd["open_from"] - hour_of_day) * 60 - minutes + \
                                    (scd["open_from"] < hour_of_day) * (24 - hour_of_day + scd["open_from"]) * 60 - minutes
                vehicle.dispatch(
                    self._get_station_coordinates(station),
                    t + minutes_till_open
                )

    def relocate_vehicle(self, vehicle_type, origin, destination):
        """ Relocate a vehicle form one station to another. The vehicle must be available and
            will remain available, but just from a different location.

        Parameters
        ----------
        vehicle_type: str, one of ['TS', 'RV', 'HV', 'WO']
            The type of vehicle that should be relocated.
        origin: str
            The name of the station from which a vehicle should be moved.
        destination: str
            The name of the station the vehicle should be moved to.
        """
        new_coords = self._get_station_coordinates(destination)
        # select vehicle
        options = [v for v in self.vehicles.values() if v.available and
                   v.current_station == origin and v.type == vehicle_type]
        try:
            options[0].relocate(destination, new_coords)
        except IndexError:
            raise ValueError("There is no vehicle available at station {} of type {}."
                             " List of options (should be empty): {}"
                             .format(origin, vehicle_type, options))

    def _prepare_results(self):
        """ Create pd.DataFrame with descriptive column names of logged results."""
        self.results = pd.DataFrame(self.log[0:self.log_index, :], columns=self.log_columns,
                                    dtype=object)

        # cast types
        dtypes = [np.float, pd.Timestamp, str, str, np.int, str, str, str,
                  np.float, np.float, np.float, np.float, np.float, np.float, str, str]

        self.results = self.results.astype(
            dtype={self.log_columns[c]: dtypes[c] for c in range(len(self.log_columns))})

        # add reponse time targets
        self.results["target"] = 10*60

    def initialize_without_simulating(self, N=100000):
        self._initialize_log(N)
        self.vehicles = self._create_vehicle_dict(self.vehicle_allocation)
        self.isampler.reset_time()
        self.t = 0

    def _get_target(self, incident_type, object_function):
        """ Get the response time norm for a given incident. """
        if incident_type in ['Binnenbrand', 'Buitenbrand']:
            return self.target_dict[object_function] * 60
        else: 
            return 15 * 60

    def simulate_single_incident(self):
        """ Simulate a random incident and its corresponding deployments.

        Requires the Simulator to be initialized with `initialize_without_simulating`.
        Simulates a single incidents and all deployments that correspond to it.
        """
        # sample incident and update status of vehicles at new time t
        self.t, time, type_, loc, prio, req_vehicles, func, dest = self._sample_incident()

        self._update_vehicles(self.t, time)

        # sample dispatch time
        dispatch = self.rsampler.sample_dispatch_time(type_)

        # get target response time
        target = self._get_target(type_, func)

        # sample rest of the response time and log everything
        for v in req_vehicles:

            vehicle, estimated_time = self._pick_vehicle(loc, v)
            if vehicle is None:
                turnout, travel, onscene, response = [np.nan]*4
                self._log([self.t, time, type_, loc, prio, func, v, "EXTERNAL", dispatch,
                           turnout, travel, onscene, response, target, "EXTERNAL", "EXTERNAL"])
            else:
                turnout, travel, onscene = self.rsampler.sample_response_time(
                    type_, loc, vehicle.current_station, vehicle.type,
                    estimated_time=estimated_time)

                vehicle.dispatch(dest, self.t + (onscene/60) + (estimated_time/60))

                response = dispatch + turnout + travel
                self._log([self.t, time, type_, loc, prio, func, vehicle.type, vehicle.id,
                           dispatch, turnout, travel, onscene, response, target,
                           vehicle.current_station, vehicle.base_station])

    def simulate_n_incidents(self, N, restart=True):
        """ Simulate N incidents and their reponses.

        Parameters
        ----------
        N: int
            The number of incidents to simulate.
        restart: boolean, optional
            Whether to empty the log and reset time before simulation (True)
            or to continue where stopped (False). Optional, defaults to True.
        """
        if restart:
            self.initialize_without_simulating()
        else:
            # keep t as it is and keep log intact
            # self.t = self.log[0:self.log_index, 0].max()
            pass

        for _ in range(N):
            self.simulate_single_incident()

        self._prepare_results()
        print("Simulated {} incidents. See Simulator.results for the results.".format(N))

    def _simulate_single_period(self):
        """ Simulate a specific period a single time.

        Simulates a period specified by Simulator.
        Returns
        -------
        results: pd.DataFrame,
            The log of simulated incidents and deployments.
        """
        self.initialize_without_simulating()
        # T = (pd.to_datetime(self.end_time) - pd.to_datetime(self.start_time)).seconds / 60
        while self.t < self.isampler.T * 60:
            self.simulate_single_incident()

        self._prepare_results()
        return self.results.copy()

    def simulate_period(self, start_time=None, end_time=None, n=1):
        """ Simulate a specific time period.

        Parameters
        ----------
        start_time: Timestamp or str, opional (default: None)
            The start time of the simulation. Must be somewhere in the interval with which
            the simulator is initialized. If None, uses the start time with which the Simulator
            is initialized.
        end)time: Timestamp or str, opional (default: None)
            The end time of the simulation. Must be somewhere in the interval with which
            the simulator is initialized. If None, uses the end time with which the Simulator
            is initialized.
        n: int, optional (default: 1)
            The number of runs, i.e., how many times to simulate the period.

        Notes
        -----
        - The simulation is started from a "base" state for all n simulation runs. This means
        all vehicles are available and at their base stations.
        - An additional column (`experiment`) is added to the log/results, denoting the number
        of the experiment.
        """
        if (start_time is not None) or (end_time is not None):
            self.set_simulation_period(start_time, end_time)
            print("Simulation period changed to {} till {}.".format(self.start_time, self.end_time))

        logs = []
        for i in range(n):
            print("\rSimulating period: {} - {}. Run: {}/{}"
                  .format(self.start_time, self.end_time, i + 1, n), end="")
            log = self._simulate_single_period()
            log["experiment"] = i + 1
            logs.append(log)

        self.results = pd.concat(logs, axis=0, ignore_index=True)
        print("\nDone. See Simulator.results for the log.")

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

        self.log_columns = ["t", "time", "incident_type", "location", "priority",
                            "object_function", "vehicle_type", "vehicle_id", "dispatch_time",
                            "turnout_time", "travel_time", "on_scene_time", "response_time",
                            "target", "station", "base_station_of_vehicle"]

        self.log = np.empty((size, 16), dtype=object)
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
        file_name: str, optional
            How to name the csv of the log. Default: 'simulation_results.csv'.
        """
        self.results.to_csv(os.path.join(self.data_dir, file_name), index=False)

    def save_simulator_object(self, path=None):
        """ Save the Simulator instance as a pickle for quick loading.

        Saves the entire Simulator object as a pickle, so that it can be quickly loaded
        with all preprocessed attributes. Note: generator objects are not supported by
        pickle, so they have to be removed before dumping and re-initialized after
        loading. Therefore, always load the simulator with fdsim.helpers.quick_load_simulator.

        Parameters
        ----------
        path: str, optional
            Where to save the file.

        Notes
        -----
        Requires the pickle package to be installed.
        """
        del self.rsampler.dispatch_generators
        del self.rsampler.turnout_generators
        del self.rsampler.travel_time_noise_generators
        del self.rsampler.onscene_generators
        del self.isampler.incident_time_generator

        if path is None:
            path = os.path.join(self.data_dir, "simulator.pickle")

        pickle.dump(self, open(path, "wb"))
        self.rsampler._create_response_time_generators()
        self.isampler.reset_time()

    def set_vehicle_allocation(self, vehicle_allocation):
        """ Assign custom allocation of vehicles to stations.

        Parameters
        ----------
        vehicle_allocation: pd.DataFrame
            The allocation of vehicles to stations. column names are
            vehicle types and row names (index) are station names. Alternatively,
            there is a column called 'kazerne' that specifies the station names.
            In the latter case, the index is ignored.
        """
        vehicle_allocation = self._preprocess_vehicle_allocation(vehicle_allocation)
        self.vehicle_allocation = vehicle_allocation.copy()
        self.vehicles = self._create_vehicle_dict(vehicle_allocation)

    def set_vehicles(self, station_name, vehicle_type, number):
        """ Set the number of vehicles for a specific station and vehicle_type.

        Parameters
        ----------
        station_name: str,
            The name of the station to change the vehicles for.
        vehicle_type: str,
            The type of vehicle to change the number of vehicles for.
        number: int
            The new number of vehicles of vehicle_type to assign to the station.
        """
        vehicle_allocation = self.vehicle_allocation.copy()
        if "kazerne" in vehicle_allocation.columns:
            vehicle_allocation.set_index("kazerne", inplace=True)
        vehicle_allocation.loc[station_name, vehicle_type] = number
        self.set_vehicle_allocation(vehicle_allocation)

    def set_station_locations(self, station_locations, vehicle_allocation, station_names=None):
        """ Assign custom locations of fire stations.

        Parameters
        ----------
        station_locations: array-like of strings
            The demand locations that should get a fire station.
        vehicle_allocation: pd.DataFrame
            The vehicles to assign to each station. Every row corresponds to
            a station in the same order as station_locations and station_names.
            Expects the column names to match the vehicle types (default:
            ["TS", "RV", "HV", "WO"]). Other columns are ignored, including 'kazerne',
            since 'station_names' is used to define the names of the stations.
        station_names: array-like of strings, optional
            The custom names of the stations. If None, will use 'STATION 1', 'STATION 2', etc.
        """
        assert len(station_locations) == len(vehicle_allocation), \
            ("Length of station_locations does not match number of rows of vehicle_allocation."
             " station_locations has length {}, while vehicle_allocation has shape {}"
             .format(len(station_locations), vehicle_allocation.shape))

        if station_names is None:
            station_names = ["STATION " + str(i) for i in range(len(station_locations))]

        self.rsampler.set_custom_stations(station_locations, station_names)
        self.dispatcher.set_custom_stations(station_locations, station_names)

        vehicle_allocation["kazerne"] = station_names
        self.set_vehicle_allocation(vehicle_allocation)

        if self.verbose: print("Custom station locations set.")

    def move_station(self, station_name, new_location, keep_name=True, keep_distributions=True, new_name=None):
        """ Move an existing station to a new location.

        Parameters
        ----------
        station_name: str
            The name of the station to move.
        new_location: str
            The identifier of the demand location to move the station to or
            the decimal longitude and latitude coordinates to move the station
            to.
        keep_name: boolean, optional (default: True)
            Whether to keep the current name of the station or not.
        keep_distribution: boolean, optional (default: True)
            Whether to keep distributions of fitted random variables, such as
            the distribution of the turn-out time.
        new_name: str, required if keep_name=False
            New name of the station. Ignored if keep_name=True.
        """
        if keep_name:
            new_name = station_name
        else:
            assert (new_name is not None), "If keep_name=False, new_name must be specified."

        self.rsampler.move_station(station_name, new_location, new_name, keep_distributions)
        self.dispatcher.move_station(station_name, new_location, new_name)

        vehicle_allocation = self.vehicle_allocation.copy()
        if not keep_name:
            if "kazerne" not in vehicle_allocation.columns:
                vehicle_allocation.reset_index(drop=False, inplace=True)
            station_index = np.nonzero(vehicle_allocation["kazerne"].values
                                       == station_name)[0][0]
            vehicle_allocation["kazerne"].iloc[station_index] = new_name
        self.set_vehicle_allocation(vehicle_allocation)

        if self.verbose:
            print("Station moved to {} and vehicles re-initialized".format(new_location))

    def reset_stations(self):
        """ Reset station locations and names to the original stations from the data. """
        self.rsampler.reset_stations()
        self.dispatcher.reset_stations()
        self.set_vehicle_allocation(self.original_vehicle_allocation)

    def set_start_time(self, start_time):
        self.start_time = pd.to_datetime(start_time)
        self._update_period()

    def _update_period(self):
        self.isampler._set_sampling_dict(self.start_time, self.end_time)
        self.isampler.incident_time_generator = self.isampler._incident_time_generator()

    def set_end_time(self, end_time):
        self.end_time = pd.to_datetime(end_time)
        self._update_period()

    def reset_simulation_period(self):
        self.isampler._set_sampling_dict(None, None)
        self.isampler.incident_time_generator = self.isampler._incident_time_generator()
        self.start_time = self.isampler.sampling_dict[0]["time"]
        self.end_time = self.isampler.sampling_dict[self.isampler.T - 1]["time"]

    def set_simulation_period(self, start_time, end_time):
        """ Change the start and end times of the simulation period. """
        if start_time is not None:
            self.start_time = pd.to_datetime(start_time)
        if end_time is not None:
            self.end_time = pd.to_datetime(end_time)
        if (start_time is not None) or (end_time is not None):
            self._update_period()
        else:
            raise ValueError("Both start and end time are None values. "
                             "Provide at least one of them.")

    def set_daily_closing_time(self, station, closed_from, open_from, remove_previous=True):
        """ Close a station every day for a specified time period.

        Parameters
        ----------
        station: str,
            The name of the station to close.
        closed_from: int,
            The hour of the day from which the station is closed.
        open_from: int,
            The hour of the day from which the station is open.
        remove_previous: boolean, optional (default: True)
            Whether to reset previously set any closing times.
        """
        if remove_previous:
            self.remove_station_closing_time(station)

        if open_from > closed_from:
            closing_hours = np.arange(closed_from, open_from, 1)
            for h in closing_hours:
                self.station_closed_dict[station][h] = True

        elif closed_from > open_from:
            closing_hours = list(np.arange(0, open_from, 1)) + list(np.arange(closed_from, 24, 1))
            for h in closing_hours:
                self.station_closed_dict[station][h] = True

        else:
            raise ValueError("Open and closing time cannot be the same. Got open at {} "
                             "and close at {}.".format(open_from, closed_from))

        self.station_closed_dict[station]["open_from"] = open_from
        self.station_closed_dict[station]["closed_from"] = closed_from
        self.station_closed_dict[station]["length_closing_period"] = len(closing_hours)

    def remove_station_closing_time(self, station):
        self.station_closed_dict[station] = {i: False for i in range(24)}

    def reset_closing_times(self):
        self.station_closed_dict = {st: {i: False for i in range(24)} for st in self.vehicle_allocation["kazerne"]}

    def evaluate_performance(self, metric="on_time", vehicles=None, priorities=None,
                             group_by=None, by_incident=True):
        """ Evaluate the performance of a finished simulation run.

        Parameters
        ----------
        metric: str, one of {'on_time', 'mean_response_time', 'mean_lateness'}, optional
            The performance metric to calculate. The available metrics are defined as follows:
            - on_time_deployments: the proportion of deployments that arrived within the
                given norm/target.
            - mean_response_time: the mean response time in seconds.
            - mean_lateness: the mean time in seconds that incidents were too late
                (deployments that were on time have a lateness of zero).

            Defaults to "on_time".
        vehicles: str or array-like of strings, optional
            The vehicle types to take into account when calculating the metric.
        priorities: int or array-like of integers in [1,3], optional
            The incident priorities to include when calculating the metric.
        group_by: str or array-like of strings, optional
            The columns to group by when calculating the performance metric.

        Returns
        -------
        A float, representing the calculated metric if group_by is None. A pd.DataFrame with
        columns [group_by[0], ..., groupby[n], <metric>], where <metric> is the name of the
        metric and the values in that column are the scores per group.
        """
        def calc_on_time(data):
            return np.mean(data["response_time"] <= data["target"])

        def calc_mean_response_time(data):
            return data["response_time"].mean()

        def calc_mean_lateness(data):
            return np.mean(np.maximum(0, data["response_time"] - data["target"]))

        # Process input variations
        if isinstance(vehicles, str):
            vehicles = [vehicles]
        if isinstance(priorities, int):
            priorities = [priorities]

        # Set evaluation function
        if metric == "on_time":
            func = calc_on_time
        elif metric == "mean_response_time":
            func = calc_mean_response_time
        elif metric == "mean_lateness":
            func = calc_mean_lateness
        else:
            raise ValueError("'metric' must be one of "
                             "['on_time', 'mean_response_time', 'mean_lateness'].")

        # Filter data
        results_filtered = self.results.copy()
        if vehicles is not None:
            results_filtered = results_filtered[np.isin(results_filtered["vehicle_type"],
                                                        vehicles)]
        if priorities is not None:
            results_filtered = results_filtered[np.isin(results_filtered["priority"],
                                                        priorities)]

        if by_incident:
            # only use first arriving TS, ignore if no TS deployed
            results_filtered = results_filtered[results_filtered["vehicle_type"] == "TS"]
            results_filtered = (results_filtered.sort_values("response_time")
                                                .groupby("t", as_index=False)
                                                .first()
                                                .dropna())
        # Calculate metric (by group)
        if group_by is not None:
            performance = (results_filtered.groupby(group_by)
                                           .apply(func)
                                           .reset_index()
                                           .rename(columns={0: metric}))
        else:
            performance = func(results_filtered)

        return performance
