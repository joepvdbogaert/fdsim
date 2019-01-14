import numpy as np
import copy


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

    def __init__(self, id_, vehicle_type, base_station_name, coords=None):
        self.id = id_
        self.type = vehicle_type
        self.current_station_name = base_station_name
        self.base_station_name = base_station_name
        self.available = True
        self.base_coords = coords
        self.last_station_name = base_station_name
        self.last_station_coords = self.base_coords
        self.coords = coords
        self.becomes_available = 0

    def assign_base_station(self, station):
        """ Assign a FireStation object to the Vehicle as a base station. """
        self.base_station = station

    def available_for_deployment(self):
        """ Check whether the vehicle is available for deployment.

        If the vehicles is at its base location, we need to check if there is a crew
        available at that station. If the vehicle is at another location, it already has
        a crew with it, so it is available if it is not already in deployment (i.e., when it
        is relocated).
        """
        if self.available_at_base():
            return np.sum(self.base_station.get_crews(self.type)) > 0
        else:
            return self.available

    def assign_crew(self):
        if self.is_at_base():
            self.current_crew = self.base_station.dispatch_crew(self.type)

    def dispatch(self, coords, t_available):
        self.last_station_coords = self.coords
        self.coords = coords
        self.available = False
        self.becomes_available = t_available

    def return_to_last_station(self):
        """ Return to last known station. """
        if self.current_station_name == self.base_station_name:
            self.return_to_base()
        else:
            # just change coordinates back to the currently assigned station
            # and set availability to True
            self.available = True
            self.coords = self.last_station_coords

    def return_to_base(self):
        self.current_station_name = self.base_station_name
        self.coords = self.base_coords
        self.last_station_coords = self.coords
        self.available = True
        self.base_station.return_crew(self.type, self.current_crew)
        self.current_crew = None

    def relocate(self, station_name, coords):
        self.assign_crew()
        self.current_station_name = station_name
        self.coords = coords

    def is_at_base(self):
        return self.current_station_name == self.base_station_name

    def available_at_base(self):
        return (self.current_station_name == self.base_station_name) * self.available


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


class FireStation():
    """ A fire station where vehicles and crews are positioned when not dispatched.

    Parameters
    ----------
    name: str,
        The name of the station.
    coords: tuple(float, float),
        Longitude and latitude of the station.
    base_vehicles: array-like,
        The fdsim.object.Vehicle objects that have this station as their base.
    crew_dict: dict,
        The crews that are available at this station. Dictionary like
        {'vehicle type' -> [fulltime crews, parttime crews]}.
    """
    crew_map = {"TS": "TS",
                "RV": "RVHV",
                "HV": "RVHV",
                "WO": "WO"}

    def __init__(self, name, coords, base_vehicles, crew_dict):
        self.name = name
        self.coords = coords
        self.assign_base_vehicles(base_vehicles)
        self.crews = copy.deepcopy(crew_dict)
        self.normal_crews = copy.deepcopy(crew_dict)
        self.status_table = np.ones((7, 24)) * 2 # days by hours, 2 for normal operations
        self.operating_status = 2 # operating normally
        self.backup_protocol = False
        self.has_status_cycle = False

    def assign_base_vehicles(self, base_vehicles):
        self.base_vehicles = base_vehicles
        self.base_vehicle_dict = {vtype: [v.id for v in base_vehicles if v.type == vtype] for
                                  vtype in np.unique([v.type for v in base_vehicles])}
        self.base_vtypes = list(self.base_vehicle_dict.keys())
        self.base_vehicle_ids = [v.id for v in base_vehicles]

    def set_status(self, day, hour, status):
        self.status_table[day, hour] = status
        self.has_status_cycle = True

    def reset_status_cycle(self):
        self.status_table = np.ones((7, 24)) * 2
        self.has_status_cycle = False

    def get_crews(self, vtype):
        return self.crews[self.crew_map[vtype]]

    def get_normal_crews(self, vtype):
        return self.normal_crews[self.crew_map[vtype]]

    def set_crews(self, vtype, ft, pt):
        self.crews[self.crew_map[vtype]] = np.array([ft, pt])

    def add_crews(self, vtype, ft, pt):
        self.crews[self.crew_map[vtype]] += np.array([ft, pt])

    def dispatch_crew(self, vtype):
        """ Assign a crew to a dispatched vehicle for deployment or relocation. The crew
        is no longer available at the station after it is dispatched. """
        ft, pt = self.get_crews(vtype)
        if ft > 0:
            self.add_crews(vtype, -1, 0)
            return "fulltime"
        elif pt > 0:
            self.add_crews(vtype, 0, -1)
            return "parttime"
        else:
            raise ValueError("No crews available for this vehicle type at this station.")

    def return_crew(self, vtype, crew_type):
        """ Return the crew from a returning vehicle to the station. """
        if crew_type == "fulltime":
            self.crews[self.crew_map[vtype]][0] += 1
        elif crew_type == "parttime":
            self.crews[self.crew_map[vtype]][1] += 1
        else:
            raise ValueError("'crew_type' must be one of ['fulltime', 'parttime']. "
                             "Received: {}".format(crew_type))

    def get_status(self, day, hour):
        return self.status_table[day, hour]

    def update_crew_status(self, day, hour):
        """ Update the available crews according to the current status of the station.

        Stations can have cycles of statuses:
        0: closed, no vehicles or crew available
        1: station operating in part time mode (i.e., normal full time crew is now part time)
        2: normal

        This method checks what the current status is given the day of the week and hour of
        day and updates the available crews accordingly.

        Parameters
        ----------
        day: int in [0, 6]
            The day number in zero-indexed integers (Monday = 0, Sunday = 6).
        hour: int in [0, 23]
            The hour of day (rounded down / ignoring minutes) in zero-indexed integers.
        """
        if self.has_status_cycle:
            status = self.get_status(day, hour)
            if status == 2:
                # (return to) normal operation
                # see what crews are deployed or relocated to other stations
                deployed_vehicles = [v for v in self.base_vehicles if not v.available_at_base()]
                for vtype in self.base_vtypes:
                    ft = len([v for v in deployed_vehicles if (v.type == vtype) and
                              (v.current_crew == "fulltime")])
                    pt = len([v for v in deployed_vehicles if (v.type == vtype) and
                              (v.current_crew == "parttime")])
                    normal_ft, normal_pt = self.get_normal_crews(vtype)
                    self.set_crews(vtype, normal_ft - ft, normal_pt - pt)

            elif status == 1:
                # everything to parttime
                deployed_vehicles = [v for v in self.base_vehicles if not v.available_at_base()]
                for vtype in self.base_vtypes:
                    ft = len([v for v in deployed_vehicles if (v.type == vtype) and
                              (v.current_crew == "fulltime")])
                    pt = len([v for v in deployed_vehicles if (v.type == vtype) and
                              (v.current_crew == "parttime")])
                    normal_ft, normal_pt = self.get_normal_crews(vtype)
                    self.set_crews(vtype, 0, normal_ft - ft + normal_pt - pt)

            elif status == 0:
                # close the station
                for vtype in self.base_vtypes:
                    self.set_crews(vtype, 0, 0)

            else:
                raise ValueError("'status' should be one of [0, 1, 2]. Got: {}"
                                 .format(status))

    def activate_backup_protocol(self):
        self.backup_protocol = True
