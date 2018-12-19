import os
import pandas as pd
import numpy as np
import warnings
import copy
from itertools import chain

from fdsim.incidentfitting import (
    get_prio_probabilities_per_type,
    get_vehicle_requirements_probabilities,
    get_spatial_distribution_per_type,
    get_building_function_probabilities
)

from fdsim.predictors import ProphetIncidentPredictor

from fdsim.responsetimefitting import (
    prepare_data_for_response_time_analysis,
    get_osrm_distance_and_duration,
    add_osrm_distance_and_duration,
    get_coordinates_locations_stations,
    model_travel_time_per_vehicle,
    fit_dispatch_times,
    fit_turnout_times,
    fit_onscene_times
)

from fdsim.objects import DemandLocation, IncidentType


class ResponseTimeSampler():
    """ Class that samples response times for deployed vehicles.

    Parameters
    ----------
    load_data: boolean
        Whether to load data from disk (True) or pre-process from scratch (False).
    data_dir: str
        The path to the directory where data is stored. Used if load_data==True
        or when data is saved after preparation.
    verbose: boolean
        Whether to print progress updates when doing stuff.
    """

    def __init__(self, load_data=True, data_dir="/data",
                 verbose=True):
        """ Initialize variables. """
        self.fitted = False
        self.data = None
        self.file_name = "response_data.csv"
        self.data_dir = data_dir
        self.verbose = verbose

        if load_data:
            try:
                self.data = pd.read_csv(os.path.join(self.data_dir, self.file_name),
                                        dtype={"hub_vak_bk": int}, low_memory=False)
                self.data["hub_vak_bk"] = self.data["hub_vak_bk"].astype(str)
            except FileNotFoundError:
                warnings.warn("No prepared data found, check if 'data_dir' specifies"
                              " the right directory. If you didn't mean to "
                              "load data from disk, initialize with 'load_data=False'."
                              " Instance intialized anyway, without data. 'fit()' method"
                              " now needs OSRM API access to prepare the data. "
                              "Given directory: {}.".format(self.data_dir))

    def fit(self, incidents=None, deployments=None, stations=None,
            vehicle_types=["TS", "RV", "HV", "WO"], osrm_host="http://192.168.56.101:5000",
            save_prepared_data=False, location_col="hub_vak_bk",
            volunteer_stations=["DRIEMOND", "DUIVENDRECHT", "AMSTELVEEN VRIJWILLIG"]):
        """ Fit random variables related to response time.

        Parameters
        ----------
        incidents: pd.DataFrame
            The incident data. Only required when no prepared data is loaded.
        deployments: pd.DataFrame (optional)
            The deployment data. Only required when no prepared data is loaded.
        stations: pd.DataFrame (optional)
            The station information including coordinates and station names.
            Only required when no prepared data is loaded.
        vehicle_types: array-like of strings
            The types of vehicles to use. Defaults to ["TS", "RV", "HV", "WO"].
        osrm_host: str
            The url to the OSRM API, required when object is initialized with
            load_data=False or when no prepared data was found.
        save_prepared_data: boolean
            Whether to write the preprocessed data to a csv file so that it can
            be loaded the next time. Defaults to False.
        location_col: str
            The name of the column that specifies the demand locations, defaults
            to "hub_vak_bk".
        volunteer_stations: array-like of str, optional (default: None)
            The names of the stations that are run by volunteers. Turn-out times
            are fitted separately for these stations, since volunteers have to travel
            to the station first.

        Notes
        -----
        Performs the following steps:
            - Prepares data (merges and adds OSRM distance and duration per
              deployment)
            - Fits lognormal random variables to dispatch times per incident type.
            - Fits Gamma random variables to turnout time per station and type.
            - Models the travel time as :math:`\alpha + \beta * \gamma (\theta, k) * \hat{t}`,
              per vehicle type. Here :math:`\hat{t}` represents the OSRM estiamte of the
              travel time and :math:`\gamma` is a random noise factor.
            - Saves the station and demand location coordinates in dictionaries.
        """
        self.location_col = location_col

        if self.data is None:
            if incidents is not None and deployments is not None and stations is not None:
                if self.verbose: print("No data loaded, start pre-processing with OSRM...")
                self._prep_data_for_fitting(incidents=incidents, deployments=deployments,
                                            stations=stations, vehicles=vehicle_types,
                                            osrm_host=osrm_host, save=save_prepared_data)
            else:
                raise ValueError("No prepared data loaded and not all data fed to 'fit()'.")

        if self.verbose: print("Extracting coordinates of stations and demand locations...")
        self.location_coords, self.station_coords = \
            get_coordinates_locations_stations(self.data, location_col=location_col)

        if self.verbose: print('Fitting random variables on response time...')
        self.high_prio_data = (self.data[(self.data["dim_prioriteit_prio"] == 1) &
                                         (self.data["inzet_terplaatse_volgnummer"] == 1)]
                               .copy())
        self.dispatch_rv_dict = fit_dispatch_times(self.high_prio_data)
        self.turnout_time_rv_dict = fit_turnout_times(self.data,
                                                      volunteer_stations=volunteer_stations)
        self.travel_time_dict = model_travel_time_per_vehicle(self.high_prio_data)
        self.onscene_time_rv_dict = fit_onscene_times(self.data)

        if self.verbose: print("Creating response time generators...")
        self._create_response_time_generators()

        if self.verbose: print("Response time variables fitted.")
        self.fitted = True

    def _prep_data_for_fitting(self, incidents, deployments, stations,
                               vehicles, osrm_host, save):
        """ Perform basic preprocessing and calculate OSRM estimates for travel time.

        Prepared data is stored under self.data. Nothing is returned.

        Parameters
        ----------
        incidents: pd.DataFrame
            The incident data.
        deployments: pd.DataFrame
            The deployment data.
        stations: pd.DataFrame
            The station information including coordinates and station names.
        vehicles: array-like of strings
            The types of vehicles to use. Defaults to ["TS", "RV", "HV", "WO"].
        osrm_host: str
            The url to the OSRM API.
        save: boolean
            Whether to save the data to a csv file after preparing it.
        """
        if self.verbose: print("Preprocessing and merging datasets...")
        data = prepare_data_for_response_time_analysis(incidents, deployments,
                                                       stations, vehicles)

        if self.verbose: print("Adding OSRM distance and duration...")
        self.data = add_osrm_distance_and_duration(data, osrm_host=osrm_host)

        if save:
            if self.verbose: print("Saving file...")
            self.data.to_csv(os.path.join(self.data_dir, self.file_name), index=False)

        if self.verbose: print("Data prepared for fitting.")

    def set_custom_stations(self, station_locations, station_names,
                            location_col="hub_vak_bk"):
        """ Change the locations of stations to custom demand locations.

        Parameters
        ----------
        station_locations: array-like of strings
            Location IDs of the custom stations, must match values in location_col
            of the objects data.
        location_col: str, optional
            Name of the column to use as a location identifier for incidents.
            Defaults to "hub_vak_bk".
        """
        assert self.fitted, "You first have to 'fit()' before setting custom stations."
        assert len(station_locations) == len(station_names), \
            ("Lengths of station_locations and station_names do not match")

        # set station coordinates
        self.station_coords = dict()
        for i in range(len(station_locations)):
            self.station_coords[station_names[i]] = \
                self.location_coords[station_locations[i]]

    def move_station(self, station_name, new_location, new_name, keep_distributions):
        """ Move the location of a single station.

        Parameters
        ----------
        station_name: str
            The name of the station to move.
        new_location: str or tuple(float, float)
            The new location of the station. If a string is passed, it is interpreted
            as the identifier of the demand location to move the station to. If a tuple of
            floats is passed, it is interpreted as the new coordinates in decimal (long, lat).
        new_name: str
            The new name of the station.
        keep_distributions: boolean
            If True, keep original turn-out time distribution. If False, use distribution
            of all turn-out times (not station-dependent).
        """
        if isinstance(new_location, tuple):
            self.station_coords[new_name] = new_location
        elif isinstance(new_location, str):
            self.station_coords[new_name] = self.location_coords[new_location]
        else:
            raise ValueError("new_location cannot be interpreted. Pass either a tuple of "
                             "decimal longitude and latitude or a string representing "
                             "a demand location.")

        if keep_distributions:
            self.turnout_time_rv_dict[new_name] = copy.deepcopy(
                self.turnout_time_rv_dict[station_name])
        else:
            turnout_dict = fit_turnout_times(self.high_prio_data, types_only=True)
            self.turnout_time_rv_dict[new_name] = turnout_dict

        if new_name != station_name:
            del self.station_coords[station_name]
            del self.turnout_time_rv_dict[station_name]

        self._create_response_time_generators()

    def reset_stations(self):
        """ Reset the station locations and names to those obtained from the data. """
        self.location_coords, self.station_coords = \
            get_coordinates_locations_stations(self.data, location_col=self.location_col)

    def _create_response_time_generators(self):
        """ Create generator objects for every element of response time.

        Sampling efficiency suffers from a high number of calls to '.rvs()' of a
        scipy.stats frozen distribution object. Hence, we avoid calling it many times
        by using generators that call the function once every 10,000 samples.

        This function creates dictionaries of the same architecture as the dictionaries
        holding the random variables, but intead of random variables, it stores generators
        as lowest level elements.

        Example
        -------
        >>> sampler = ResponseTimeSampler(args)
        >>> # sample next dispatch time for a 'Binnenbrand'
        >>> next(sampler.dispatch_generators["Binnenbrand"])

        Notes
        -----
        It is not useful to call this function manually, it is called upon initialization
        of the ResponseTimeSampler.
        """
        def time_generator(rv):
            """ A generator of random samples according to RV (random variable). """
            a = rv.rvs(10000)
            counter = 0
            while True:
                try:
                    yield a[counter]
                    counter += 1
                except IndexError:
                    counter = 0
                    a = rv.rvs(10000)

        self.dispatch_generators = {}
        for incident_type, rv in self.dispatch_rv_dict.items():
            self.dispatch_generators[incident_type] = time_generator(rv)

        self.turnout_generators = {}
        for appointment in self.turnout_time_rv_dict.keys():
            self.turnout_generators[appointment] = {}
            for prio, rv in self.turnout_time_rv_dict[appointment].items():
                self.turnout_generators[appointment][prio] = time_generator(rv)

        self.travel_time_noise_generators = {}
        for vehicle_type, v_dict in self.travel_time_dict.items():
            rv = v_dict["noise_rv"]
            self.travel_time_noise_generators[vehicle_type] = time_generator(rv)

        self.onscene_generators = {}
        for incident_type in self.onscene_time_rv_dict.keys():
            self.onscene_generators[incident_type] = {}
            for vehicle_type, rv in self.onscene_time_rv_dict[incident_type].items():
                self.onscene_generators[incident_type][vehicle_type] = time_generator(rv)

    def sample_dispatch_time(self, incident_type):
        """ Sample a random dispatch time, given the incident type.

        Parameters
        ----------
        incident_type: str,
            The type of incident to sample dispatch times for.

        Returns
        -------
        int, the random dispatch time in seconds.
        """
        return next(self.dispatch_generators[incident_type])

    def sample_travel_time(self, estimated_time, vehicle,
                           osrm_host="http://192.168.56.101:5000"):
        """ Sample a random travel time.

        Parameters
        ----------
        estimated_time: float
            The travel time in seconds according to OSRM.
        vehicle: str
            The vehicle type (code) to sample travel time for.

        Returns
        -------
        A float representing the random travel time in seconds.
        """
        try:
            d = self.travel_time_dict[vehicle]
        except KeyError:
            d = self.travel_time_dict["overall"]

        noise = next(self.travel_time_noise_generators[vehicle])
        return d["a"] + d["b"] * noise * estimated_time

    def sample_response_time(self, incident_type, location_id, station_name, vehicle_type,
                             appointment, prio, estimated_time=None,
                             osrm_host="http://192.168.56.101:5000"):
        """ Sample a random response time based on deployment characteristics.

        Parameters
        ----------
        incident_type: str
            The type of the incident to sample turn-out time for.
        location_id: str
            The ID of the demand location where the incident takes place.
        station_name: str
            The name of the station that the deployment is executed from.
        vehicle_type: str
            The vehicle type (code) to sample travel time for.
        estimated_time: float, int, optional
            The estimated travel time according to OSRM. Optional, defaults
            to None. If None, estimation is collected from OSRM at time of
            calling, which is far less efficient.
        osrm_host: str, optional
            The URL to the OSRM API. Required when no 'estimated_time' is provided.

        Returns
        -------
        Tuple of (turn-out time, travel time, on-scene time). Note that the dispatch
        time is sampled separately, since that is only done once per incident, while
        this function is called per deployment.
        """
        if estimated_time is None:
            orig = self.station_coords[station_name]
            dest = self.location_coords[location_id]
            _, estimated_time = get_osrm_distance_and_duration(orig, dest, osrm_host=osrm_host)

        turnout = next(self.turnout_generators[appointment][prio])
        travel = self.sample_travel_time(estimated_time, vehicle_type)
        onscene = next(self.onscene_generators[incident_type][vehicle_type])
        return turnout, travel, onscene


class IncidentSampler():
    """ Samples timing and details of incidents from distributions in the data.

    Parameters
    ----------
    incidents: pd.DataFrame
        The incident data to obtain distributions from.
    deployments: pd.DataFrame
        The deployments to obtain vehicle requirement distributions from.
    vehicles: array-like of strings
        The vehicles types to include in the sampling.
    start_time: datetime or str (convertible to datetime) or None
        The start of the period that should be simulated. If none, starts from
        end of the data. Defaults to None.
    end_time: datetime or str (convertible to datetime) or None
        The start of the period that should be simulated. If none, ends one year
        after end of the data. Defaults to None.
    predictor: str, one of ['prophet']
        What incident rate forecast method to use. Currently only supports
        "prophet" (based on Facebook's Prophet package).
    verbose: boolean
        Whether to print progress.

    Example
    -------
    >>> sampler = IncidentSampler(df_incidents, df_deployments)
    >>> # sample 10 incidents and print details
    >>> t = 0
    >>> for _ in range(10):
    >>>     t, type_, loc, prio, vehicles, func = sampler.sample_next_incident(t)
    >>>     print("time: {}, type: {}, loc: {}, prio: {}, vehicles: {}, object function: {}."
    >>>           .format(t, type_, loc, prio, vehicles, func))
    """

    def __init__(self, incidents, deployments, vehicle_types, locations, start_time=None,
                 end_time=None, predictor="prophet", fc_dir="/data", verbose=True):
        """ Initialize all properties by extracting probabilities from the data. """
        self.incidents = incidents[
                np.in1d(incidents["hub_vak_bk"].fillna(0).astype(int).astype(str), locations)]
        self.deployments = deployments
        self.vehicle_types = vehicle_types
        self.verbose = verbose

        self.types = self._infer_incident_types()

        self._assign_predictor(predictor, fc_dir)
        self._set_sampling_dict(start_time, end_time, incident_types=self.types)
        self._create_incident_types()
        self._create_demand_locations()

        self.incident_time_generator = self._incident_time_generator()

        if self.verbose: print("IncidentSampler ready for simulation.")

    def _infer_incident_types(self):
        """ Create list of incident types based on provided data. """
        merged = pd.merge(self.deployments[["hub_incident_id", "voertuig_groep"]],
                          self.incidents[["dim_incident_id", "dim_incident_incident_type"]],
                          left_on="hub_incident_id", right_on="dim_incident_id", how="left")
        merged = merged[np.isin(merged["voertuig_groep"], self.vehicle_types)]
        return [v for v in merged["dim_incident_incident_type"].unique()
                if str(v) not in ["nan", "NVT"]]

    def _set_sampling_dict(self, start_time, end_time, incident_types=None):
        """ Get the dictionary required for sampling from the predictor.

        Gets the sampling dictionary of a Predictor and stores it to the
        IncidentSampler object so that it can be used in sampling. Also
        stores the length of the dictionary as self.T. This is used during
        simulation to loop over the sampling dictionary while avoiding
        IndexErrors.

        Parameters
        ----------
        start_time: timestamp, str convertible to timestamp, or None
            The start date and time of the period to simulate. If None,
            use all available timestamps in the forecast.
        end_time: timestamp, str convertible to timestamp, or None
            The start date and time of the period to simulate. If None,
            use all available timestamps in the forecast.
        incident_types: array-like of strings, optional (default: None)
            The incident types to incorporate in the simulation. if None,
            uses inferred incident types.
        """
        if incident_types is None:
            incident_types = self.types

        self.sampling_dict = self.predictor.create_sampling_dict(start_time, end_time,
                                                                 incident_types)
        self.T = len(self.sampling_dict)
        self.lambdas = 60 / np.array([d['beta'] for d in self.sampling_dict.values()])

    def _assign_predictor(self, predictor, fc_dir):
        """ Initialize incident rate predictor and assign to property.

        Parameters
        ----------
        predictor: str, one of ['prophet']
            The predictor to use to forecast the incident rates. Currently,
            only supports predictor='prophet'.
        """
        if predictor == "prophet":
            if self.verbose: print("Initializing incident rate predictor...")
            self.predictor = ProphetIncidentPredictor(load_forecast=True, fc_dir=fc_dir,
                                                      verbose=True)
        else:
            raise ValueError("Currently, only predictor='prophet' is supported.")

    def _create_incident_types(self):
        """ Initialize incident types with their characteristics.

        Creates a dictionary of IncidentType objects. Every such object holds
        type-specific distributions about priority, required vehicles,
        and demand locations.
        """
        if self.verbose: print("Getting priority probabilities..")
        prio_probs = get_prio_probabilities_per_type(self.incidents)

        if self.verbose: print("Getting vehicle requirement probabilities...")
        vehicle_probs = get_vehicle_requirements_probabilities(self.incidents,
                                                               self.deployments,
                                                               self.vehicle_types)

        if self.verbose: print("Getting spatial distributions...")
        location_probs = get_spatial_distribution_per_type(self.incidents)

        if self.verbose: print("Initializing incident types...")
        self.incident_types = {t: IncidentType(prio_probs[t], vehicle_probs[t],
                               location_probs[t]) for t in self.types}

    def _create_demand_locations(self):
        """ Initialize demand locations and their building function distributions.

        Creates a dictionary of DemandLocation objects. Each such object has its
        own distribution over building functions that is used during sampling.
        """
        if self.verbose: print("Getting building function probabilities...")
        building_probs = get_building_function_probabilities(self.incidents)

        if self.verbose: print("Initializing demand locations")
        self.locations = {l: DemandLocation(l, building_probs[l])
                          for l in building_probs.keys()}

    def _incident_time_generator(self, period_length=60):
        """ Returns a generator object for incident times. """

        counter = 0

        while True:

            past_time = counter * len(self.lambdas) * period_length

            # process periods in sampling dict in one go
            n_arrivals = np.random.poisson(self.lambdas, size=len(self.lambdas))
            total_arrivals = np.sum(n_arrivals)

            periods = np.array([x for x in chain.from_iterable(
                                [[i]*n_arrivals[i] for i in range(len(n_arrivals))])],
                               dtype=int)

            minutes = np.random.uniform(0, period_length, size=total_arrivals)
            times = np.sort(periods * period_length + minutes)

            # yield times one by one
            for time in times:
                yield time + past_time

            counter += 1

    def reset_time(self):
        """ Reset the incident time generator to start from t=0. """
        self.incident_time_generator = self._incident_time_generator()

    def _sample_incident_details(self, incident_type):
        """ Draw random sample of incident characteristics.

        Parameters
        ----------
        incident_type: str
            The type of incident as occuring in the data.

        Returns
        -------
        Tuple of (location id, priority, Tuple(required vehicle types), building function)
        """
        ty = self.incident_types[incident_type]
        vehicles = ty.sample_vehicles()
        priority = ty.sample_priority()
        location = ty.sample_location()
        building_function = self.locations[location].sample_building_function(incident_type)

        return location, priority, vehicles, building_function

    def sample_next_incident(self):
        """ Sample a random time and type for the next incident.

        Parameters
        ----------
        t: float
            The current time in minutes (from an arbitrary random start time)

        Returns
        -------
        The new time t of the next incident and incident details:
        (t, incident_type, location, priority, vehicles, building function)
        """
        t = next(self.incident_time_generator)
        d = self.sampling_dict[int((t//60) % self.T)]
        incident_type = np.random.choice(a=self.types, p=d["type_distribution"])
        loc, prio, veh, func = self._sample_incident_details(incident_type)

        return t, d["time"], incident_type, loc, prio, veh, func
