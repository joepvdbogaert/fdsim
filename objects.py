import numpy as np


class Vehicle():
    """ A vehicle of the fire department.

    Parameters
    ----------
    id: int or str
        Identifier for this vehicle instance.
    vehicle_type: str
        The type of vehicle, one of 'TS', 'RV', 'HV', 'WO'.
    base_station: str
        The name of the station this vehicle belongs to.
    coords: Tuple(float, float), optional
        The decimal longitude and latitude showing the current location
        of the vehicle.
    """

    def __init__(self, id_, vehicle_type, base_station, coords=None):
        self.id = id_
        self.type = vehicle_type
        self.current_station = base_station
        self.base_station = base_station
        self.available = True
        self.base_coords = coords
        self.coords = coords
        self.becomes_available = 0

    def dispatch(self, coords, t_available):
        self.coords = coords
        self.available = False
        self.becomes_available = t_available

    def return_to_base(self):
        self.current_station = self.base_station
        self.coords = self.base_coords
        self.available = True

    def relocate(self, station, coords):
        self.current_station = station
        self.coords = coords


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
