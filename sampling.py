import os
import pandas as pd
import numpy as np
import warnings

from incidentfitting import (
    get_prio_probabilities_per_type,
    get_vehicle_requirements_probabilities,
    get_spatial_distribution_per_type,
    get_building_function_probabilities
)

from predictors import ProphetIncidentPredictor

from responsetimefitting import (
    prepare_data_for_response_time_analysis,
    get_osrm_distance_and_duration,
    add_osrm_distance_and_duration,
    get_coordinates_locations_stations,
    model_travel_time_per_vehicle,
    fit_dispatch_times,
    fit_turnout_times
)

from objects import DemandLocation, IncidentType
from definitions import ROOT_DIR


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

    def __init__(self, load_data=True, data_dir="data/responsetimes",
                 verbose=True):
        """ Initialize variables. """
        self.fitted = False
        self.data = None
        self.file_name = "response_data.csv"
        self.data_dir = os.path.join(ROOT_DIR, data_dir)
        self.verbose = verbose

        if load_data:
            try:
                self.data = pd.read_csv(os.path.join(self.data_dir, self.file_name),
                                        dtype={"hub_vak_bk": int})
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
            save_prepared_data=False, location_col="hub_vak_bk"):
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
        osrm_host: str
            The url to the OSRM API, required when object is initialized with
            load_data=False or when no prepared data was found.
        save_prepared_data: boolean
            Whether to write the preprocessed data to a csv file so that it can
            be loaded the next time. Defaults to False.
        location_col: str
            The name of the column that specifies the demand locations, defaults
            to "hub_vak_bk".

        Notes
        -----
        Performs the following steps:
            - Prepares data (merges and adds OSRM distance and duration per
              deployment)
            - Fits lognormal random variables to dispatch times per incident type.
            - Fits Gamma random variables to turnout time per station and type.
            - Models the travel time as $\alpha + \beta * \gamma (\theta, k) * \hat{t},
              per vehicle type. Here $\hat{t}$ represents the OSRM estiamte of the
              travel time and $\gamma$ is a random noise factor.
            - Saves the station and demand location coordinates in dictionaries.
        """

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
        self.high_prio_data = self.data[(self.data["dim_prioriteit_prio"] == 1) &
                                        (self.data["inzet_terplaatse_volgnummer"] == 1)]
        self.dispatch_rv_dict = fit_dispatch_times(self.high_prio_data)
        self.turnout_time_rv_dict = fit_turnout_times(self.high_prio_data)
        self.travel_time_dict = model_travel_time_per_vehicle(self.high_prio_data)

        if self.verbose: print("Response time variables fitted.")
        self.fitted = True

    def _prep_data_for_fitting(self, incidents, deployments, stations, vehicles, osrm_host, save):
        """ Perform basic preprocessing and calculate OSRM estimates for travel time.

        Parameters
        ----------
        incidents: pd.DataFrame
            The incident data.
        deployments: pd.DataFrame
            The deployment data.
        stations: pd.DataFrame
            The station information including coordinates and station names.
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

    def set_custom_stations(self, station_locations, location_col="hub_vak_bk"):
        """ Change the locations of stations to custom demand locations.

        Parameters
        ----------
        station_locations: array-like of strings
            Location IDs of the custom stations, must match values in location_col
            of the objects data.
        location_col: str
            Name of the column to use as a location identifier for incidents.
        """
        assert self.fitted, "You fist have to 'fit()' before setting custom stations."
        self.station_coords = dict()
        for i in range(len(station_locations)):
            self.station_coords["STATION " + str(i)] = \
                self.location_coords[station_locations[i]].copy()

        if self.verbose: print("Custom station locations set.")

    def sample_dispatch_time(self, incident_type):
        """ Sample a random dispatch time given the type of incident.

        Parameters
        ----------
        incident_type: str
            The type of the incident to sample dispatch time for.

        Returns
        -------
        A float representing the random dispatch time in seconds.
        """
        return self.dispatch_rv_dict[incident_type].rvs()

    def sample_turnout_time(self, incident_type, station_name):
        """ Sample a random dispatch time given the type of incident.

        Parameters
        ----------
        incident_type: str
            The type of the incident to sample turn-out time for.
        station_name: str
            The name of the station that the deployment is executed from.

        Returns
        -------
        A float representing the random turn-out time in seconds.
        """
        return self.turnout_time_rv_dict[station_name][incident_type].rvs()

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

        noise = d["noise_rv"].rvs()
        return d["a"] + d["b"] * noise * estimated_time

    def sample_on_scene_time(self, incident_type, vehicle_type):
        """ Sample the duration that the vehicle is on-scene at the incident. """
        return 30*60 # seconds

    def sample_response_time(self, incident_type, location_id, station_name, vehicle_type,
                             estimated_time=None, osrm_host="http://192.168.56.101:5000"):
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

        Returns
        -------
        Tuple of (dispatch time, turn-out time, travel time, on-scene time,
        total response time). Note that the first three elements add up to the total response
        time. Those are returned separately to enable storing detailed sampling results.
        """
        orig = self.station_coords[station_name]
        dest = self.location_coords[location_id]

        if estimated_time is None:
            _, estimated_time = get_osrm_distance_and_duration(orig, dest, osrm_host=osrm_host)

        dispatch = self.sample_dispatch_time(incident_type)
        turnout = self.sample_turnout_time(incident_type, station_name)
        travel = self.sample_travel_time(estimated_time, vehicle_type)
        onscene = self.sample_on_scene_time(incident_type, vehicle_type)
        response = dispatch + turnout + travel

        return dispatch, turnout, travel, onscene, response


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

    def __init__(self, incidents, deployments, vehicle_types, start_time=None, end_time=None,
                 predictor="prophet", verbose=True):
        """ Initialize all properties by extracting probabilities from the data. """
        self.incidents = incidents
        self.deployments = deployments
        self.vehicle_types = vehicle_types
        self.verbose = verbose

        self.types = self._infer_incident_types()

        self._assign_predictor(predictor)
        self._set_sampling_dict(start_time, end_time, incident_types=self.types)
        self._create_incident_types()
        self._create_demand_locations()

        if self.verbose: print("IncidentSampler ready for simulation.")

    def _infer_incident_types(self):
        """ Create list of incident types based on provided data. """
        merged = pd.merge(self.deployments[["hub_incident_id", "voertuig_groep"]],
                          self.incidents[["dim_incident_id", "dim_incident_incident_type"]],
                          left_on="hub_incident_id", right_on="dim_incident_id", how="left")
        merged = merged[np.isin(merged["voertuig_groep"], self.vehicle_types)]
        return [v for v in merged["dim_incident_incident_type"].unique()
                if str(v) not in ["nan", "NVT"]]

    def _set_sampling_dict(self, start_time, end_time, incident_types):
        """ Get the dictionary required for sampling from the predictor. """
        self.sampling_dict = self.predictor.create_sampling_dict(start_time, end_time,
                                                                 incident_types)
        self.T = len(self.sampling_dict)

    def _assign_predictor(self, predictor):
        """ Initialize incident rate predictor and assign to property. """
        if predictor == "prophet":
            if self.verbose: print("Initializing incident rate predictor...")
            self.predictor = ProphetIncidentPredictor(load_forecast=True, verbose=True)
        else:
            raise ValueError("Currently, only predictor='prophet' is supported.")

    def _create_incident_types(self):
        """ Initialize incident types with their characteristics. """
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
        """ Initialize demand locations and their building function distributions. """
        if self.verbose: print("Getting building function probabilities...")
        building_probs = get_building_function_probabilities(self.incidents)

        if self.verbose: print("Initializing demand locations")
        self.locations = {l: DemandLocation(l, building_probs[l])
                          for l in building_probs.keys()}

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

    def sample_next_incident(self, t):
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
        d = self.sampling_dict[int(t//60 % (self.T))]
        t += np.random.exponential(d["beta"])
        incident_type = np.random.choice(a=self.types, p=d["type_distribution"])
        loc, prio, veh, func = self._sample_incident_details(incident_type)

        return t, d["time"], incident_type, loc, prio, veh, func
