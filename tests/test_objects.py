from pytest import approx, fixture
import numpy as np
import pandas as pd
from fdsim.objects import Vehicle, FireStation


class TestObjectsModule(object):

    @classmethod
    def setup_class(cls):
        # cls.resource_allocation = pd.read_csv("../Data/resource_allocation.csv", sep=";", decimal=".")
        cls.resource_allocation = pd.DataFrame({
            'kazerne': {0: 'OSDORP', 1: 'TEUNIS', 2: 'IJSBRAND', 3: 'ZEBRA', 4: 'PIETER', 5: 'DIRK', 6: 'NICO', 7: 'HENDRIK',
                        8: 'WILLEM', 9: 'DIEMEN', 10: 'DUIVENDRECHT', 11: 'AMSTELVEEN', 12: 'AALSMEER', 13: 'UITHOORN', 14:
                        'DRIEMOND', 15: 'ANTON', 16: 'VICTOR'},
            'TS': {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 2, 10: 1, 11: 2, 12: 2, 13: 2, 14: 1, 15: 1, 16: 1},
            'RV': {0: 0, 1: 1, 2: 0, 3: 0, 4: 0, 5: 1, 6: 1, 7: 1, 8: 1, 9: 0, 10: 0, 11: 1, 12: 0, 13: 0, 14: 0, 15: 1, 16: 1},
            'HV': {0: 0, 1: 1, 2: 0, 3: 0, 4: 1, 5: 0, 6: 0, 7: 0, 8: 0, 9: 1, 10: 0, 11: 1, 12: 0, 13: 0, 14: 0, 15: 0, 16: 0},
            'WO': {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 1, 9: 0, 10: 0, 11: 1, 12: 0, 13: 0, 14: 0, 15: 0, 16: 0},
            'TS_crew_ft': {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 0, 11: 1, 12: 0, 13: 0, 14: 0, 15: 1, 16: 1},
            'TS_crew_pt': {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 1, 10: 1, 11: 1, 12: 2, 13: 2, 14: 1, 15: 0, 16: 0},
            'RVHV_crew_ft': {0: 1, 1: 2, 2: 1, 3: 0, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 0, 11: 1, 12: 0, 13: 0, 14: 0, 15: 1, 16: 1},
            'RVHV_crew_pt': {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0, 16: 0},
            'WO_crew_ft': {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 1, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0, 16: 0},
            'WO_crew_pt': {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 1, 12: 0, 13: 0, 14: 0, 15: 0, 16: 0}})

    @classmethod
    def teardown_class(cls):
        pass

    @fixture
    def vehicle(self):
        return Vehicle("VEHICLE 1", "TS", "VICTOR", (4.9291813, 52.360456799999994))

    @fixture
    def vehicles_amstelveen(self):
        coords = (4.8818307999999995, 52.3016997)
        vehicles = [Vehicle("VEHICLE 1", "TS", "AMSTELVEEN", coords),
                    Vehicle("VEHICLE 2", "TS", "AMSTELVEEN", coords),
                    Vehicle("VEHICLE 3", "HV", "AMSTELVEEN", coords),
                    Vehicle("VEHICLE 4", "RV", "AMSTELVEEN", coords),
                    Vehicle("VEHICLE 5", "WO", "AMSTELVEEN", coords)]

        return vehicles

    @fixture
    def coords_amstelveen(self):
        return (4.8818307999999995, 52.3016997)

    @fixture
    def crew_dict_amstelveen(self):
        rs = self.resource_allocation.copy()
        rs["kazerne"] = rs["kazerne"].str.upper()
        rs.set_index("kazerne", inplace=True)
        a = "AMSTELVEEN"
        return {"TS": np.array([rs.loc[a, "TS_crew_ft"], rs.loc[a, "TS_crew_pt"]]),
                "RVHV": np.array([rs.loc[a, "RVHV_crew_ft"], rs.loc[a, "RVHV_crew_pt"]]),
                "WO": np.array([rs.loc[a, "WO_crew_ft"], rs.loc[a, "WO_crew_pt"]])}

    @fixture
    def amstelveen(self, vehicles_amstelveen, coords_amstelveen, crew_dict_amstelveen):
        amsv = FireStation("AMSTELVEEN", coords_amstelveen, vehicles_amstelveen, crew_dict_amstelveen)
        for v in vehicles_amstelveen:
            v.assign_base_station(amsv)
        return amsv

    @staticmethod
    def dictionaries_equal(a, b):
        if set(list(a.keys())) == set(list(b.keys())):
            # if keys are identical, check values
            return np.all([a[k] == b[k] for k in a.keys()])
        else:
            # keys are not equal, so False
            return False

    def test_station_init(self, amstelveen, crew_dict_amstelveen):
        """Test whether the FireStation instance was correctly initialized."""
        assert amstelveen.name == "AMSTELVEEN", "Name is wrong"
        assert np.all(amstelveen.coords == approx((4.8818307999999995, 52.3016997))), "Coords are off"
        assert len(amstelveen.base_vehicles) == 5, "Number of vehicles is off"
        assert self.dictionaries_equal(amstelveen.crews, crew_dict_amstelveen), "Crew dict is wrong"
        assert self.dictionaries_equal(amstelveen.normal_crews, amstelveen.crews), "Crews not equal to normal_crews"
        assert np.all(amstelveen.status_table == 2.), "Status table not all '2's."
        assert amstelveen.backup_protocol == False, "backup protocol should be False"
        assert amstelveen.has_status_cycle == False, "has_status_cycle should be False"
        assert np.all(np.in1d(["TS", "RV", "HV", "WO"], amstelveen.base_vtypes)), "Not all vehicle types present"

    def test_vehicle_base_station(self, amstelveen):
        for v in amstelveen.base_vehicles:
            assert isinstance(v.base_station, FireStation)
            assert v.base_station.name == "AMSTELVEEN"

    def test_set_status_and_reset_status_cycle(self, amstelveen):
        amstelveen.set_status(4, 18, 0)
        assert amstelveen.has_status_cycle == True, "has_status_cycle not set to True"
        assert amstelveen.status_table[4, 18] == approx(0), "Status not updated in table"
        amstelveen.set_status(6, 12, 1)
        assert amstelveen.has_status_cycle == True, "has_status_cycle not set to True"
        assert amstelveen.status_table[4, 18] == approx(0), "Previous status change should not be removed"
        assert amstelveen.status_table[6, 12] == approx(1), "Status not updated in table"
        # reset
        amstelveen.reset_status_cycle()
        assert np.all(amstelveen.status_table == approx(2)), "Not correctly reset to normal"

    def test_get_crews(self, amstelveen, crew_dict_amstelveen):
        assert np.all(amstelveen.get_crews("TS") == crew_dict_amstelveen["TS"])
        assert np.all(amstelveen.get_crews("RV") == crew_dict_amstelveen["RVHV"])
        assert np.all(amstelveen.get_crews("HV") == crew_dict_amstelveen["RVHV"])
        assert np.all(amstelveen.get_crews("WO") == crew_dict_amstelveen["WO"])

    def test_get_normal_crews(self, amstelveen, crew_dict_amstelveen):
        assert np.all(amstelveen.get_normal_crews("TS") == crew_dict_amstelveen["TS"])
        assert np.all(amstelveen.get_normal_crews("RV") == crew_dict_amstelveen["RVHV"])
        assert np.all(amstelveen.get_normal_crews("HV") == crew_dict_amstelveen["RVHV"])
        assert np.all(amstelveen.get_normal_crews("WO") == crew_dict_amstelveen["WO"])

    def test_update_crews_no_deployments(self, amstelveen, crew_dict_amstelveen):
        amstelveen.set_status(3, 8, 0)
        amstelveen.update_crew_status(3, 8)
        assert amstelveen.get_status(3, 8) == approx(0.), "get_status return wrong number"
        for vtype in ["TS", "RV", "HV", "WO"]:
            assert np.all(amstelveen.get_crews(vtype) == np.zeros(2)), "Crews not set to zero"

        amstelveen.set_status(6, 7, 1)
        amstelveen.update_crew_status(6, 7)
        new_crew_dict = {"TS": [0, np.sum(crew_dict_amstelveen["TS"])],
                         "RVHV": [0, np.sum(crew_dict_amstelveen["RVHV"])],
                         "WO": [0, np.sum(crew_dict_amstelveen["WO"])]}
        assert np.all(amstelveen.get_crews("TS") == new_crew_dict["TS"])
        assert np.all(amstelveen.get_crews("RV") == new_crew_dict["RVHV"])
        assert np.all(amstelveen.get_crews("HV") == new_crew_dict["RVHV"])
        assert np.all(amstelveen.get_crews("WO") == new_crew_dict["WO"])

        # assert normal_crews is not affected
        assert np.all(amstelveen.get_normal_crews("TS") == crew_dict_amstelveen["TS"]), "normal_crews should never change"
        assert np.all(amstelveen.get_normal_crews("RV") == crew_dict_amstelveen["RVHV"]), "normal_crews should never change"
        assert np.all(amstelveen.get_normal_crews("HV") == crew_dict_amstelveen["RVHV"]), "normal_crews should never change"
        assert np.all(amstelveen.get_normal_crews("WO") == crew_dict_amstelveen["WO"]), "normal_crews should never change"

    def test_update_crews_with_deployments(self, amstelveen, crew_dict_amstelveen):
        # set a status cycle and select a vehicle
        amstelveen.set_status(3, 8, 0)
        amstelveen.set_status(6, 7, 1)
        vehicle = [v for v in amstelveen.base_vehicles if v.type == "TS"][0]

        # assign a crew and dispatch to random place for a random time
        vehicle.assign_crew()
        vehicle.dispatch((4.33, 5.22), 60.3)
        assert np.all(amstelveen.get_crews("TS") == (crew_dict_amstelveen["TS"] - np.array([1, 0]))), "Crew not dispatched correctly"

        # close station and test available crews
        amstelveen.update_crew_status(3, 8)
        for vtype in ["TS", "RV", "HV", "WO"]:
            assert np.all(amstelveen.get_crews(vtype) == np.zeros(2)), "Crews not set to zero"

        # return vehicle to base and return crew
        vehicle.return_to_base()
        assert np.all(amstelveen.crews["TS"] == [1, 0]), "Crew not returned correctly"

        # update crews (without changing actual status)
        amstelveen.update_crew_status(3, 8)
        assert np.all(amstelveen.crews["TS"] == [0, 0]), "Crews not updated correctly"

        # go directly to part time mode
        amstelveen.update_crew_status(6, 7)
        new_crew_dict = {"TS": [0, np.sum(crew_dict_amstelveen["TS"])],
                         "RVHV": [0, np.sum(crew_dict_amstelveen["RVHV"])],
                         "WO": [0, np.sum(crew_dict_amstelveen["WO"])]}
        assert np.all(amstelveen.get_crews("TS") == new_crew_dict["TS"])
        assert np.all(amstelveen.get_crews("RV") == new_crew_dict["RVHV"])
        assert np.all(amstelveen.get_crews("HV") == new_crew_dict["RVHV"])
        assert np.all(amstelveen.get_crews("WO") == new_crew_dict["WO"])

        # dispatch
        vehicle.assign_crew()
        vehicle.dispatch((4.33, 5.22), 60.3)
        amstelveen.update_crew_status(6, 7) # still part time
        assert np.all(amstelveen.get_crews("TS") == [0, np.sum(crew_dict_amstelveen["TS"]) - 1])

        # back to normal
        amstelveen.update_crew_status(0, 0) # normal
        assert amstelveen.get_status(0, 0) == 2, "This should be 2 (normal operation)"
        assert vehicle.current_crew == "parttime", "assigned crew should be part time (station was in part time mode)"
        assert np.all(amstelveen.get_crews("TS") == crew_dict_amstelveen["TS"] - np.array([0, 1])), "Wrong update"

        vehicle.return_to_base()
        assert np.all(amstelveen.get_crews("TS") == crew_dict_amstelveen["TS"]), "Wrong update"

        # dispatch with full time crew, return when in part time mode
        vehicle.assign_crew()
        vehicle.dispatch((4.33, 5.22), 60.3)
        assert vehicle.current_crew == "fulltime"
        amstelveen.update_crew_status(6, 7)
        vehicle.return_to_base()
        assert np.all(amstelveen.get_crews("TS") == [1, np.sum(crew_dict_amstelveen["TS"]) - 1])
        amstelveen.update_crew_status(6, 7)
        assert np.all(amstelveen.get_crews("TS") == [0, np.sum(crew_dict_amstelveen["TS"])])

    def test_vehicle_availability_due_to_crews(self, amstelveen):
        amstelveen.set_status(0, 0, 0)
        amstelveen.update_crew_status(0, 0)
        assert np.all([v.available_for_deployment() for v in amstelveen.base_vehicles] == [False] * len(amstelveen.base_vehicles)), "All vehicles should be unavailable"
        amstelveen.set_status(1, 1, 1)
        amstelveen.update_crew_status(1, 1)
        assert np.all([v.available_for_deployment() for v in amstelveen.base_vehicles] == [True] * len(amstelveen.base_vehicles)), "All vehicles should be available"
        amstelveen.update_crew_status(2, 2) # normal
        assert np.all([v.available_for_deployment() for v in amstelveen.base_vehicles] == [True]  * len(amstelveen.base_vehicles)), "All vehicles should be available"
        # dispatch RV and check if HV is unavailable
        hv = [v for v in amstelveen.base_vehicles if v.type == "HV"][0]
        rv = [v for v in amstelveen.base_vehicles if v.type == "RV"][0]
        hv.assign_crew()
        assert rv.available_for_deployment() == False, "Vehicle should not be available for deployment anymore"
