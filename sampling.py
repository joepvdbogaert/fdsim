import numpy as np

from incidentfitting import (
    get_prio_probabilities_per_type,
    get_vehicle_requirements_probabilities,
    get_spatial_distribution_per_type,
    get_building_function_probabilities
    )

from predictors import ProphetIncidentPredictor


class DemandLocation():
    """ An area in which incidents occur.

    Parameters
    ----------
    location_id: int
        Identifier of the demand location.
    building_probs: dictionary
        Specifies probabilities of an incident happening in
        a certain type of building. Dictonary must be defined like
        {'incident type': {'building function': prob}.
    """

    def __init__(self, location_id, building_probs):
        self.id = location_id
        self.building_probs = building_probs

    def sample_building_function(self, incident_type):
        return np.random.choice(a=list(self.building_probs[incident_type].keys()),
                                p=list(self.building_probs[incident_type].values()))


class IncidentType():
    """ A type of incident.

    Parameters
    ----------
    prio_probs: list
        list of probabilities that this type of incident is of
        priority [1, 2, 3] respectively.
    vehicle_probs: dictionary
        Combinations of vehicles are the keys and the values are the
        probabilities that this combination is required.
    location_probs: dictionary
        Keys are the location identifiers and values are the probability
        of the incident happening in that location.
    """

    def __init__(self, prio_probs, vehicle_probs, location_probs):
        self.prio_probs = prio_probs
        self.vehicle_probs = vehicle_probs
        self.location_probs = location_probs

    @staticmethod
    def random_choice(a, p):
        """ Like np.random.choice but for larger arrays. """
        return a[np.digitize(np.random.sample(), np.cumsum(p))]

    def sample_priority(self):
        """ Sample the priority of an instance of this inciden type. """
        return np.random.choice(a=[1, 2, 3], p=self.prio_probs)

    def sample_location(self):
        """ Sample the demand location for an instance of this incident type. """
        return self.random_choice(list(self.location_probs.keys()),
                                  list(self.location_probs.values()))

    def sample_vehicles(self):
        """ Sample required vehicles for an instance of this incident type. """
        return self.random_choice(list(self.vehicle_probs.keys()),
                                  list(self.vehicle_probs.values()))


class IncidentSampler():
    """ Samples timing and details of incidents from distributions in the data.

    Parameters
    ----------
    incidents: pd.DataFrame
        The incident data to obtain distributions from.
    deployments: pd.DataFrame
        The deployments to obtain vehicle requirement distributions from.
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

    def __init__(self, incidents, deployments, start_time=None, end_time=None,
                 predictor="prophet", verbose=True):
        """ Initialize all properties by extracting probabilities from the data. """
        self.incidents = incidents
        self.deployments = deployments
        self.verbose = verbose

        self.types = [t for t in self.incidents["dim_incident_incident_type"].unique()
                      if str(t) not in ["nan", "NVT"]]

        self._assign_predictor(predictor)
        self._set_sampling_dict(start_time, end_time)
        self._create_incident_types()
        self._create_demand_locations()

        if self.verbose: print("IncidentSampler ready for simulation.")

    def _set_sampling_dict(self, start_time, end_time):
        """ Get the dictionary required for sampling from the predictor. """
        self.sampling_dict = self.predictor.create_sampling_dict(start_time, end_time)
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
                                                               self.deployments)

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

        return t, incident_type, loc, prio, veh, func
