import os
import numpy as np
import pandas as pd
import pickle

from fdsim.sampling import IncidentSampler, ResponseTimeSampler
from fdsim.objects import Vehicle, FireStation
from fdsim.dispatching import ShortestDurationDispatcher
from fdsim.helpers import progress


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
    resource_allocation: pd.DataFrame
        The allocation of vehicles and crews to stations. Expected columns:
        ["kazerne", "TS", "RV", "HV", "WO", "TS_crew_ft", "TS_crew_pt", "RVHV_crew_ft",
         "RVHV_crew_pt", "WO_crew_ft", "WO_crew_pt"].
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
    >>> sim = Simulator(incidents, deployments, stations, resource_allocation)
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
    target_dict = {'Bijeenkomstfunctie': 10,
                   'Industriefunctie': 8,
                   'Woonfunctie': 8,
                   'Straat': 10,
                   'Overige gebruiksfunctie': 10,
                   'Kantoorfunctie': 10,
                   'Logiesfunctie': 8,
                   'Onderwijsfunctie': 8,
                   'Grachtengordel': 10,
                   'Overig': 10,
                   'Winkelfunctie': 5,
                   'Kanalen en rivieren': 10,
                   'nan': 10,
                   'Trein': 5,
                   'Sportfunctie': 10,
                   'Regionale weg': 10,
                   'Celfunctie': 5,
                   'Tram': 5,
                   'Metro': 5,
                   'Sloten en Vaarten': 10,
                   'Gezondheidszorgfunctie': 8,
                   'Lokale weg': 5,
                   'Polders': 10,
                   'Haven': 10,
                   'Autosnelweg': 10,
                   'Meren en plassen': 10,
                   'Hoofdweg': 10,
                   'unknown': 10}

    def __init__(self, incidents, deployments, stations, resource_allocation,
                 load_response_data=True, load_time_matrix=True, save_response_data=False,
                 save_time_matrix=False, vehicle_types=["TS", "RV", "HV", "WO"],
                 predictor="basic", start_time=None, end_time=None, data_dir="data",
                 osrm_host="http://192.168.56.101:5000", location_col="hub_vak_bk",
                 verbose=True):

        self.data_dir = data_dir
        self.verbose = verbose
        self.station_data = stations
        self.vehicle_types = vehicle_types

        progress("Start processing data.", verbose=self.verbose)
        self.resource_allocation = self._preprocess_resource_allocation(resource_allocation)
        self.original_resource_allocation = self.resource_allocation.copy()

        self.rsampler = ResponseTimeSampler(load_data=load_response_data,
                                            data_dir=self.data_dir,
                                            verbose=verbose)

        self.rsampler.fit(incidents=incidents, deployments=deployments, stations=stations,
                          vehicle_types=vehicle_types, osrm_host=osrm_host,
                          save_prepared_data=save_response_data, location_col=location_col)

        progress("Fitting incident distributions.", verbose=self.verbose)
        locations = list(self.rsampler.location_coords.keys())
        self.isampler = IncidentSampler(incidents, deployments, vehicle_types, locations,
                                        start_time=start_time, end_time=end_time,
                                        predictor=predictor,
                                        fc_dir=os.path.join(data_dir),
                                        verbose=verbose)

        self.vehicles = self._create_vehicles(self.resource_allocation)
        self.stations = self._create_stations(self.resource_allocation)
        self._add_base_stations_to_vehicles()

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

        progress("Simulator is ready. At your service.", verbose=self.verbose)

    @staticmethod
    def _preprocess_resource_allocation(resource_allocation):
        """ Preprocess the resource allocation table. """
        resource_allocation["kazerne"] = resource_allocation["kazerne"].str.upper()
        return resource_allocation

    def _create_vehicles(self, resource_allocation):
        """ Create a dictionary of Vehicle objects from the resource data.

        Parameters
        ----------
        resource_allocation: pd.DataFrame
            The allocation of resources (including vehicles) to stations. Should at least
            contain the columns ["kazerne", "TS", "RV", "HV", "WO"].Values are the number
            of vehicles assigned to the station.

        Returns
        -------
        A dictionary like: {'vehicle id' -> Vehicle object}.
        """
        vehicle_allocation = resource_allocation[["kazerne", "TS", "RV", "HV", "WO"]].copy()
        vs = vehicle_allocation.set_index("kazerne").unstack().reset_index()

        vdict = {}
        id_counter = 1

        for r in range(len(vs)):

            for _ in range(vs[0].iloc[r]):

                this_id = "VEHICLE " + str(id_counter)
                vtype = vs["level_0"].iloc[r]
                station = vs["kazerne"].iloc[r]
                coords = self._get_station_coordinates(station)

                vdict[this_id] = Vehicle(
                    this_id,
                    vtype,
                    station,
                    coords=coords,
                )

                id_counter += 1

        return vdict

    def _create_stations(self, resource_allocation):
        """ Initialize FireStation objects according to the resource allocation.

        Parameters
        ----------
        resource_allocation: pd.DataFrame,
            The resource allocation. Should at least contain the columns:
            ["TS_crew_ft", "TS_crew_pt", "RVHV_crew_ft",
             "RVHV_crew_pt", "WO_crew_ft", "WO_crew_pt"]
 
        Returns
        -------
        Stations: dict,
            A dictionary like {'station name' -> fdsim.objects.FireStation object}.
        """
        rs = resource_allocation
        station_dict = {}

        for i in range(len(rs["kazerne"])):
            station = rs["kazerne"].iloc[i]
            coords = self._get_station_coordinates(station)
            base_vehicles = [v for v in self.vehicles.values() if v.base_station_name == station]
            crew_dict = {"TS": np.array([rs["TS_crew_ft"].iloc[i], rs["TS_crew_pt"].iloc[i]]),
                         "RVHV": np.array([rs["RVHV_crew_ft"].iloc[i], rs["RVHV_crew_pt"].iloc[i]]),
                         "WO": np.array([rs["WO_crew_ft"].iloc[i], rs["WO_crew_pt"].iloc[i]])}

            station_dict[station] = FireStation(station, coords, base_vehicles, crew_dict)

        return station_dict

    # def assign_backup_crew(self, station):
    #     """ Let the part time crew at a station come to the station when the full time crew
    #     is dispatched in order to minimize response times for a second incident.

    #     Parameters
    #     ----------
    #     station: str,
    #         The station for which to use the backup protocol.

    #     Notes
    #     -----
    #     Make sure to provide the station name in all capitalized letters (e.g., 'VICTOR').
    #     Currently only looks at TS type vehicles.
    #     """
    #     assert self.station_appointments[station] == "mixed", \
    #         "This method is only applicable to stations with both full and part time crews."
    #     assert len(self.base_vehicles[station]["TS"]) > 1, \
    #         "Station must have at least 2 TS vehicles"

    #     tss = [self.vehicles[vid] for vid in self.base_vehicles[station]["TS"]]
    #     fulltime_ts = [ts.id for ts in tss if ts.base_appointment == "fulltime"][0]
    #     parttime_ts = [ts.id for ts in tss if ts.base_appointment == "parttime"][0]
    #     self.vehicles[parttime_ts].assign_as_backup_for(fulltime_ts)

    def _add_base_stations_to_vehicles(self):
        """ After initializing stations and vehicles, assign FireStation objects to Vehicles
        and the other way around.
        """
        for vehicle in self.vehicles.values():
            vehicle.assign_base_station(self.stations[vehicle.base_station_name])

        for station in self.stations.values():
            station.assign_base_vehicles([v for v in self.vehicles.values() if
                                          v.base_station_name == station.name])

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
        candidates = [v for v in self.vehicles.values() if (v.type == vehicle_type)
                      and v.available_for_deployment()]

        vehicle_id, estimated_time = self.dispatcher.dispatch(location, candidates)

        if vehicle_id == "EXTERNAL":
            return None, None

        vehicle = self.vehicles[vehicle_id]

        return vehicle, estimated_time

    @staticmethod
    def _calc_minutes_till_event(t, hour_of_day, hour_of_event):
        """ Calculate the number of minutes till a certain hour of day.

        Parameters
        ----------
        t: float,
            Minutes since start of simulation, assuming the simulation started at
            some O'clock time.
        hour_of_event: int,
            The hour of day of the event to calculate the time to.

        Returns
        -------
        time: float,
            The time in minutes from now (t) till the event.
        """
        minutes = t % 60
        return (hour_of_event > hour_of_day)*(hour_of_event - hour_of_day)*60 - minutes + \
               (hour_of_event < hour_of_day)*(24 - hour_of_day + hour_of_event)*60 - minutes

    def _update_vehicles(self, t, time):
        """ Return vehicles that finished their jobs to their base stations.

        Parameters
        ----------
        t: float
            The time since the start of the simulation. Determines which vehicles
            have become available again and can return to their bases.
        """
        # return vehicles from deployments
        for vehicle in self.vehicles.values():
            if (not vehicle.available) and (vehicle.becomes_available <= t):
                crew_type = vehicle.current_crew
                vehicle.return_to_last_station()

        # update station status and crew availability
        for station in self.stations.values():
            station.update_crew_status(time.weekday(), time.hour)

        # send relocated vehicles back home if original vehicle is available
        for vehicle in self.vehicles.values():
            # available and at other station: see if it should go home now
            if (vehicle.available) and not (vehicle.is_at_base()):
                base_vs = self.stations[vehicle.current_station_name].base_vehicle_dict[vehicle.type]
                # send vehicles back to base recursively
                if np.sum([self.vehicles[v].available_at_base() for v in base_vs]) > 0:
                    self._send_vehicle_to_base_recursively(vehicle.id)

    def _return_vehicle_to_station(self, vehicle_id, to_base=True, station=None):
        """ Send a vehicle back to the station. """
        pass

    def _send_vehicle_to_base_recursively(self, vehicle_id):
        """ Send a vehicle to it's base and send interim vehicles that may have relocated
        there to their bases. Repeat recursively.
        """
        vehicle = self.vehicles[vehicle_id]
        vehicle.return_to_base()

        # find other vehicles at station that can return to base
        relocated_vehicles = [v for v in self.vehicles.values() if
                              (v.current_station_name == vehicle.base_station_name) and
                              (v.type == vehicle.type) and
                              (not v.is_at_base())]

        for v in relocated_vehicles:
            v.return_to_base()

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
        options = [v for v in self.vehicles.values() if v.available_for_deployment() and
                   (v.current_station_name == origin) and (v.type == vehicle_type)]
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
                  np.float, np.float, np.float, np.float, np.float, np.float, str, str, str]

        self.results = self.results.astype(
            dtype={self.log_columns[c]: dtypes[c] for c in range(len(self.log_columns))})

    def initialize_without_simulating(self, N=100000):
        self._initialize_log(N)
        self.vehicles = self._create_vehicles(self.resource_allocation)
        self.stations = self._create_stations(self.resource_allocation)
        self._add_base_stations_to_vehicles()
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
                           turnout, travel, onscene, response, target, "EXTERNAL", "EXTERNAL", "EXTERNAL"])
            else:
                vehicle.assign_crew()

                turnout, travel, onscene = self.rsampler.sample_response_time(
                    type_, loc, vehicle.current_station_name, vehicle.type, vehicle.current_crew,
                    prio, estimated_time=estimated_time)

                vehicle.dispatch(dest, self.t + (onscene/60) + (estimated_time/60))

                response = dispatch + turnout + travel
                self._log([self.t, time, type_, loc, prio, func, vehicle.type, vehicle.id,
                           dispatch, turnout, travel, onscene, response, target,
                           vehicle.current_station_name, vehicle.base_station_name, vehicle.current_crew])

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

        for _ in range(N):
            self.simulate_single_incident()

        self._prepare_results()
        progress("Simulated {} incidents. See Simulator.results for the results.".format(N))

    def _simulate_single_period(self):
        """ Simulate a specific period a single time.

        Simulates a period specified by Simulator.start_time and Simulator.end_time.

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

    def simulate_period(self, start_time=None, end_time=None, n=1, restart=True):
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
        - An additional column (`run`) is added to the log/results, denoting the number
        of the run/experiment.
        """
        if (start_time is not None) or (end_time is not None):
            self.set_simulation_period(start_time, end_time)
            print("Simulation period changed to {} till {}."
                  .format(self.start_time, self.end_time))

        # continue with higher run number if we don't restart
        if restart:
            first_run = 1
        else:
            first_run = int(self.results["run"].max() + 1)
            previous_results = self.results.copy()
            progress("Continue simulation at run {}.".format(first_run))
        last_run = int(first_run + n - 1)

        # loop over runs
        logs = []
        for i in range(first_run, last_run + 1):
            progress("Simulating period: {} - {}. Run: {}/{}"
                     .format(self.start_time, self.end_time, i, last_run),
                     same_line=True, newline_end=(i == last_run))
            # execute run and save corresponding log
            log = self._simulate_single_period()
            log["run"] = i
            logs.append(log)

        if restart:
            self.results = pd.concat(logs, axis=0, ignore_index=True)
        else:
            self.results = pd.concat([previous_results] + logs, axis=0, ignore_index=True)

        progress("Done. See Simulator.results for the log.")

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
                            "target", "station", "base_station_of_vehicle", "crew_type"]

        self.log = np.empty((size, 17), dtype=object)
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
            progress("Log full at: {} entries, log extended.".format(self.log_index))
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

    def set_resource_allocation(self, resource_allocation):
        """ Assign custom allocation of vehicles to stations.

        Parameters
        ----------
        resource_allocation: pd.DataFrame
            The allocation of vehicles and crews to stations. column names are
            vehicle types and row names (index) are station names. Alternatively,
            there is a column called 'kazerne' that specifies the station names.
            In the latter case, the index is ignored.
        """
        resource_allocation = self._preprocess_resource_allocation(resource_allocation)
        self.resource_allocation = resource_allocation.copy()
        self.vehicles = self._create_vehicles(self.resource_allocation)
        self.stations = self._create_stations(self.resource_allocation)
        self._add_base_stations_to_vehicles()

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
        resource_allocation = self.resource_allocation.copy()
        if "kazerne" in resource_allocation.columns:
            resource_allocation.set_index("kazerne", inplace=True)
        resource_allocation.loc[station_name, vehicle_type] = number
        resource_allocation.reset_index(inplace=True)
        self.set_resource_allocation(resource_allocation)

    def set_crews(self, station_name, vehicle_type, number, appointment="ft"):
        """ Set the number of vehicles for a specific station and vehicle_type.

        Parameters
        ----------
        station_name: str,
            The name of the station to change the vehicles for.
        vehicle_type: str,
            The type of vehicle to change the number of vehicles for.
        number: int
            The new number of vehicles of vehicle_type to assign to the station.
        appointment: str,
            One of ['ft', "pt"] for full time or part time crew respectively.
        """
        resource_allocation = self.resource_allocation.copy()
        if "kazerne" in resource_allocation.columns:
            resource_allocation.set_index("kazerne", inplace=True)
        crew_col = vehicle_type + "_crew_" + appointment
        resource_allocation.loc[station_name, crew_col] = number
        resource_allocation.reset_index(inplace=True)
        self.set_resource_allocation(resource_allocation)

    def set_station_locations(self, station_locations, resource_allocation, station_names=None):
        """ Assign custom locations of fire stations.

        Parameters
        ----------
        station_locations: array-like of strings
            The demand locations that should get a fire station.
        resource_allocation: pd.DataFrame
            The vehicles and crews to assign to each station. Every row corresponds to
            a station in the same order as station_locations and station_names.
            Expects the column names to match the vehicle types (default:
            ["TS", "RV", "HV", "WO"]) and crew types. Other columns are ignored,
            including 'kazerne', since 'station_names' is used to define the names
            of the stations.
        station_names: array-like of strings, optional
            The custom names of the stations. If None, will use 'STATION 1', 'STATION 2', etc.
        """
        assert len(station_locations) == len(resource_allocation), \
            ("Length of station_locations does not match number of rows of resource_allocation."
             " station_locations has length {}, while resource_allocation has shape {}"
             .format(len(station_locations), resource_allocation.shape))

        if station_names is None:
            station_names = ["STATION " + str(i) for i in range(len(station_locations))]

        self.rsampler.set_custom_stations(station_locations, station_names)
        self.dispatcher.set_custom_stations(station_locations, station_names)

        resource_allocation["kazerne"] = station_names
        self.set_resource_allocation(resource_allocation)

        progress("Custom station locations set.")

    def move_station(self, station_name, new_location, keep_name=True, new_name=None):
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
        new_name: str, required if keep_name=False
            New name of the station. Ignored if keep_name=True.
        """
        if keep_name:
            new_name = station_name
        else:
            assert (new_name is not None), "If keep_name=False, new_name must be specified."

        self.rsampler.move_station(station_name, new_location, new_name)
        self.dispatcher.move_station(station_name, new_location, new_name)

        resource_allocation = self.resource_allocation.copy()
        if not keep_name:
            station_index = np.nonzero(resource_allocation["kazerne"].values
                                       == station_name)[0][0]
            resource_allocation["kazerne"].iloc[station_index] = new_name
        self.set_resource_allocation(resource_allocation)

        progress("Station moved to {} and vehicles re-initialized".format(new_location))

    def reset_stations(self):
        """ Reset station locations and names to the original stations from the data. """
        self.rsampler.reset_stations()
        self.dispatcher.reset_stations()
        self.set_resource_allocation(self.original_resource_allocation)

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

    def set_daily_station_status(self, station_name, start_hour, end_hour,
                                 days_of_week=[0, 1, 2, 3, 4, 5, 6], status="closed",
                                 remove_previous=False):
        """ Close or operate a station in part time specified hours every week.

        Parameters
        ----------
        station_name: str,
            The name of the station to close.
        start_hour: int,
            The hour of the day from which the station is closed.
        end_hour: int,
            The hour of the day from which the station is open.
        days_of_week: array-like, optional (default: [0, 1, 2, 3, 4, 5, 6]),
            The days of the week for which the status adjustment applies in zero-based
            integers (i.e., Monday = 0, Tuesday = 1, ..., Sunday = 6).
        status: str, one of ['closed', 'parttime'], optional (default: 'closed'),
            Whether the station should be completely closed or operating as a part time
            station during the specified hours.
        remove_previous: boolean, optional (default: False)
            Whether to reset previously set any closing times.

        Notes
        -----
        To set a certain status for the whole day(s), set start_hour and end_hour to the
        same value. It does not matter what value this is.
        """
        assert status in ["closed", "parttime"], \
            "Status must be one of {}".format(["closed", "parttime"])
        status_num = 0 if status == "closed" else 1

        if remove_previous:
            self.remove_station_status_cycle(station_name)

        if end_hour > start_hour:
            hours = np.arange(start_hour, end_hour, 1)
        elif start_hour > end_hour:
            hours = list(np.arange(0, end_hour, 1)) + list(np.arange(start_hour, 24, 1))
        else:
            hours = np.arange(0, 24, 1)

        for day in days_of_week:
            for h in hours:
                self.stations[station_name].set_status(day, h, status_num)
        progress("Set status of {} to {} (code: {}) for hours {} on days {}"
                 .format(station_name, status, status_num, hours, days_of_week))

    def remove_station_status_cycle(self, station_name):
        self.stations[station_name].reset_status_cycle()

    def reset_all_station_status_cycles(self):
        for station in self.stations.values():
            station.reset_status_cycle()

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
