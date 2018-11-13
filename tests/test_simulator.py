from pytest import approx
import numpy as np
import pandas as pd
from fdsim.helpers import quick_load_simulator
import fdsim


class TestSimulator(object):

    @classmethod
    def setup_class(cls):
        cls.deployments = pd.read_csv("../Data/inzetten_2008-heden.csv", sep=";", decimal=",", low_memory=False)
        cls.incidents = pd.read_csv("../Data/incidenten_2008-heden.csv", sep=";", decimal=",", low_memory=False)
        cls.stations = pd.read_excel("../Data/kazernepositie en voertuigen.xlsx", sep=";", decimal=".")
        cls.vehicle_allocation = pd.read_csv("../Data/voertuigenallocatie.csv", sep=";", decimal=".",
                                             usecols=["kazerne", "#TS", "#RV", "#HV", "#WO"], nrows=19)
        print("start loading sim")
        cls.sim = quick_load_simulator()

    @classmethod
    def teardown_class(cls):
        del cls.sim

    # @pytest.fixture
    # def sim(self):
    #     return quick_load_simulator()

    def test_quick_load(self):
        assert isinstance(self.sim, fdsim.simulation.Simulator), \
            "Loaded simulator not an instance of fdsim.simulation.Simulator"

    def test_create_vehicle_dict_normal(self):
        klass = type(self)
        vehicles = klass.sim._create_vehicle_dict(klass.vehicle_allocation)
        assert isinstance(vehicles, dict), "Expected a dictionary"
        assert klass.vehicle_allocation.drop("kazerne", axis=1).sum().sum() == len(vehicles), "Number of vehicles does not match"
        assert len([v for v in vehicles.values() if v.type == "TS"]) == klass.vehicle_allocation["TS"].sum(), "Incorrect number of TS vehicles"
        assert len([v for v in vehicles.values() if v.type == "RV"]) == klass.vehicle_allocation["RV"].sum(), "Incorrect number of RV vehicles"
        assert len([v for v in vehicles.values() if v.type == "HV"]) == klass.vehicle_allocation["HV"].sum(), "Incorrect number of HV vehicles"
        assert len([v for v in vehicles.values() if v.type == "WO"]) == klass.vehicle_allocation["WO"].sum(), "Incorrect number of WO vehicles"
        for i in range(len(klass.vehicle_allocation)):
            station_name = klass.vehicle_allocation["kazerne"].iloc[i]
            assert len([v for v in vehicles.values() if v.base_station == station_name]) == klass.vehicle_allocation.set_index("kazerne").iloc[i, :].sum(), \
                "Incorrect number of vehicles for station {}".format(station_name)

    def test_create_vehicle_dict_empty(self):
        empty_allocation = self.vehicle_allocation.copy()
        for col in ["TS", "HV", "RV", "WO"]:
            empty_allocation[col] = 0
            print(empty_allocation)

        vehicles = self.sim._create_vehicle_dict(empty_allocation)
        assert vehicles == dict(), "Expected emtpy dictionary"

    def test_create_vehicle_dict_ts_only(self):
        allocation = self.vehicle_allocation.copy()
        for col in ["HV", "RV", "WO"]:
            allocation[col] = 0

        vehicles = self.sim._create_vehicle_dict(allocation)
        assert np.all(np.unique([v.type for v in vehicles.values()]) == np.array(["TS"])), "Expected only TS vehicles"
        assert len([v for v in vehicles.values()]) == len([v for v in vehicles.values() if v.type == "TS"]), \
            "Expected all vehicles to be of type TS"

    def test_move_station_and_reset(self):
        station = "AMSTELVEEN"
        new_loc = "13710056"
        self.sim.move_station("AMSTELVEEN", "13710056", keep_name=True, keep_distributions=True)
        assert np.all(self.sim._get_station_coordinates(station) == self.sim._get_coordinates(new_loc)), \
            "Coordinates of station not updated correctly"
        assert len(self.sim.rsampler.station_coords.keys()) == 19, "Length of station coordinates is off."

        station_index = np.nonzero(self.sim.dispatcher.station_names == station)[0][0]
        loc_index = np.nonzero(self.sim.dispatcher.matrix_names == new_loc)[0][0]
        assert (self.sim.dispatcher.time_matrix_stations[station_index, :] == self.sim.dispatcher.time_matrix[loc_index, :]).all(), \
            "Time matrix not updated correctly"
        for coords in [tuple(v.coords) for v in self.sim.vehicles.values() if v.current_station == station]:
            assert coords == self.sim._get_coordinates(new_loc), "Coordinate of vehicles not updated correctly"

        self.sim.reset_stations()
        assert (self.sim.dispatcher.time_matrix_stations[station_index, :] == self.sim.dispatcher.time_matrix_df.loc[station, :].values).all(), \
            "Time matrix not reset correctly."

    def test_set_vehicles(self):
        station_name = "OSDORP"
        vehicle_type = "TS"
        number = 5
        original_number = self.sim.vehicle_allocation.set_index("kazerne").loc[station_name, vehicle_type]
        assert self.sim.vehicle_allocation.set_index("kazerne").loc[station_name, vehicle_type] == original_number, \
            "Vehicle allocation not initialized correctly"
        assert len([v for v in self.sim.vehicles.values() if v.base_station == station_name and v.type == vehicle_type]) == original_number, \
            "Vehicle objects not initialized correctly"

        self.sim.set_vehicles(station_name, vehicle_type, number)
        assert self.sim.vehicle_allocation.set_index("kazerne").loc[station_name, vehicle_type] == number, \
            "Vehicle allocation not updated correctly"
        assert len([v for v in self.sim.vehicles.values() if v.base_station == station_name and v.type == vehicle_type]) == number, \
            "Vehicle objects not updated correctly"

        station_name_2 = "AMSTELVEEN"
        vehicle_type_2 = "WO"
        number_2 = 2
        self.sim.set_vehicles(station_name_2, vehicle_type_2, number_2)
        assert self.sim.vehicle_allocation.set_index("kazerne").loc[station_name_2, vehicle_type_2] == number_2, \
            "Vehicle allocation not updated correctly"
        assert len([v for v in self.sim.vehicles.values() if v.base_station == station_name_2 and v.type == vehicle_type_2]) == number_2, \
            "Vehicle objects not updated correctly"
        assert self.sim.vehicle_allocation.set_index("kazerne").loc[station_name, vehicle_type] == number, \
            "Vehicle allocation not updated correctly"
        assert len([v for v in self.sim.vehicles.values() if v.base_station == station_name and v.type == vehicle_type]) == number, \
            "Vehicle objects not updated correctly"

        self.sim.reset_stations()
        assert self.sim.vehicle_allocation.set_index("kazerne").loc[station_name, vehicle_type] == original_number, \
            "Vehicle allocation not initialized correctly"
        assert len([v for v in self.sim.vehicles.values() if v.base_station == station_name and v.type == vehicle_type]) == original_number, \
            "Vehicle objects not initialized correctly"
        self.sim.reset_stations()

    def test_log_result(self):
        n = 100
        self.sim.simulate_n_incidents(n, restart=True)
        assert isinstance(self.sim.results, pd.DataFrame)
        assert self.sim.results.shape[1] == 16, "Expected results dataframe of shape (100+, 16), got {}".format(self.sim.results.shape)
        assert "target" in self.sim.results.columns, "Expected 'target' column to be added to columns upon finishing simulation"
        assert self.sim.results.drop(["turnout_time", "travel_time", "on_scene_time", "response_time"], axis=1).isnull().sum().sum() == 0, "Unexpected missing values in results"
        assert self.sim.results["t"].nunique() == n, "Expected 100 unique values for 't'"
        assert self.sim.results.groupby("t")["dispatch_time"].nunique().mean() == 1, \
            "Expected dispatch times to be the same for deployments belonging to the same incident"
        assert self.sim.results.groupby("t")["dispatch_time"].nunique().max() == 1, \
            "Expected dispatch times to be the same for deployments belonging to the same incident"
        not_null = self.sim.results.dropna()
        assert np.allclose(not_null["dispatch_time"] + not_null["turnout_time"] + not_null["travel_time"], not_null["response_time"]), \
            "Response times not equal to sum of its elements for all deployments"

    def test_evaluate_performance_allmetrics_all_deployments(self):
        self.sim.simulate_n_incidents(10, restart=True)
        self.sim.results = self.sim.results.iloc[0:10, :].copy()
        self.sim.results["vehicle_type"] = ["TS", "RV", "TS", "WO", "HV", "TS", "HV", "TS", "WO", "RV"]
        r = np.array([450, 230, 350, 790, 359, 206, 383, 625, 893, 634], dtype=float)
        self.sim.results["response_time"] = r
        self.sim.results["target"] = np.array([600] * 10, dtype=float)
        ontime_score = self.sim.evaluate_performance(metric="on_time", vehicles=None, priorities=None, group_by=None, by_incident=False)
        meanresponsetime_score = self.sim.evaluate_performance(metric="mean_response_time", vehicles=None, priorities=None, group_by=None, by_incident=False)
        meanlateness_score = self.sim.evaluate_performance(metric="mean_lateness", vehicles=None, priorities=None, group_by=None, by_incident=False)
        assert ontime_score == approx(np.mean(r <= 600)), "On-time proportion not as expected. Got {}, expected: {}".format(ontime_score, np.mean(r <= 600))
        assert meanresponsetime_score == approx(np.mean(r)), "Mean response time not as expected"
        assert meanlateness_score == approx(np.mean(np.maximum(0, r-600))), "Mean lateness not as expected"

    def test_evaluate_performance_allmetrics_ts_only(self):
        self.sim.simulate_n_incidents(10, restart=True)
        self.sim.results = self.sim.results.iloc[0:10, :].copy()
        self.sim.results["vehicle_type"] = ["TS", "RV", "TS", "WO", "HV", "TS", "HV", "TS", "WO", "RV"]
        r = np.array([450, 230, 350, 790, 359, 206, 383, 625, 893, 634], dtype=float)
        self.sim.results["response_time"] = r
        self.sim.results["target"] = np.array([600] * 10, dtype=float)
        ts_responses = self.sim.results["response_time"].values[self.sim.results["vehicle_type"] == "TS"]
        ontime_score = self.sim.evaluate_performance(metric="on_time", vehicles=["TS"], priorities=None, group_by=None, by_incident=False)
        meanresponsetime_score = self.sim.evaluate_performance(metric="mean_response_time", vehicles=["TS"], priorities=None, group_by=None, by_incident=False)
        meanlateness_score = self.sim.evaluate_performance(metric="mean_lateness", vehicles=["TS"], priorities=None, group_by=None, by_incident=False)
        assert ontime_score == approx(0.75), "On-time proportion not as expected, got {}".format(ontime_score)
        assert meanresponsetime_score == approx(np.mean(ts_responses)), "Mean response time not as expected. Got {}".format(meanresponsetime_score)
        assert meanlateness_score == approx(25/4), "Mean lateness not as expected. Got {}".format(meanlateness_score)

    def test_evaluate_performance_allmetrics_groupby_type(self):
        self.sim.simulate_n_incidents(10, restart=True)
        self.sim.results = self.sim.results.iloc[0:10, :].copy()
        self.sim.results["vehicle_type"] = ["TS", "RV", "TS", "WO", "HV", "TS", "HV", "TS", "WO", "RV"]
        r = np.array([450, 230, 350, 600, 359, 206, 383, 625, 893, 634], dtype=float)
        self.sim.results["response_time"] = r
        self.sim.results["target"] = np.array([600] * 10, dtype=float)
        ontime_scores = self.sim.evaluate_performance(metric="on_time", vehicles=None, priorities=None, group_by="vehicle_type", by_incident=False)
        meanresponsetime_scores = self.sim.evaluate_performance(metric="mean_response_time", vehicles=None, priorities=None, group_by="vehicle_type", by_incident=False)
        meanlateness_scores = self.sim.evaluate_performance(metric="mean_lateness", vehicles=None, priorities=None, group_by="vehicle_type", by_incident=False)

        assert ontime_scores.shape == (4, 2), "Output for on_time has unexpected shape: {}".format(ontime_scores.shape)
        assert meanresponsetime_scores.shape == (4, 2), "Output for mean_response_timehas unexpected shape: {}".format(meanresponsetime_scores.shape)
        assert meanlateness_scores.shape == (4, 2), "Output for mean_lateness has unexpected shape: {}".format(meanlateness_scores.shape)

        ts = np.array([450, 350, 206, 625])
        rv = np.array([230, 634])
        wo = np.array([600, 893])
        hv = np.array([359, 383])

        ts_ontime, ts_mean, ts_meanlate = np.mean(ts <= 600), np.mean(ts), 25 / 4
        rv_ontime, rv_mean, rv_meanlate = np.mean(rv <= 600), np.mean(rv), 34 / 2
        wo_ontime, wo_mean, wo_meanlate = np.mean(wo <= 600), np.mean(wo), 293 / 2
        hv_ontime, hv_mean, hv_meanlate = np.mean(hv <= 600), np.mean(hv), 0

        ontimes = [hv_ontime, rv_ontime, ts_ontime, wo_ontime]
        means = [hv_mean, rv_mean, ts_mean, wo_mean]
        lates = [hv_meanlate, rv_meanlate, ts_meanlate, wo_meanlate]

        assert list(ontime_scores.columns) == ["vehicle_type", "on_time"], "Columns for on_time have unexpected names"
        assert list(meanresponsetime_scores.columns) == ["vehicle_type", "mean_response_time"], "Columns for mean_response_time have unexpected names"
        assert list(meanlateness_scores.columns) == ["vehicle_type", "mean_lateness"], "Columns for mean_lateness have unexpected names"

        assert np.allclose(ontime_scores.sort_values("vehicle_type")["on_time"].values, ontimes), \
            "Values for on_time are off. Expected {}, got {}.".format(
            ontimes, ontime_scores.sort_values("vehicle_type")["on_time"].values)

        assert np.allclose(meanresponsetime_scores.sort_values("vehicle_type")["mean_response_time"].values, means), \
            "Values for on_time are off. Expected {}, got {}.".format(
            means, meanresponsetime_scores.sort_values("vehicle_type")["mean_response_time"].values)

        assert np.allclose(meanlateness_scores.sort_values("vehicle_type")["mean_lateness"].values, lates), \
            "Values for on_time are off. Expected {}, got {}.".format(
            lates, meanlateness_scores.sort_values("vehicle_type")["mean_lateness"].values)

    def test_evaluate_performance_allmetrics_by_incident_combined_with_filter_and_group_by_on_priority(self):
        fake_results = pd.DataFrame({"t": [12.1, 12.1, 36.3, 36.3, 36.3, 120.2, 152.8, 152.8, 152.8, 152.8],
                                     "vehicle_type": ["TS", "RV", "HV", "TS", "TS", "HV", "TS", "WO", "TS", "RV"],
                                     "response_time": [589, 454, 324, 623, 701, 210, 569, 1204, 890, 340],
                                     "priority": [1, 1, 2, 2, 2, 3, 1, 1, 1, 1],
                                     "times": [100]*10})

        self.sim.simulate_n_incidents(10, restart=True)
        self.sim.results = self.sim.results.iloc[0:10, :].copy()
        self.sim.results["t"] = fake_results["t"]
        self.sim.results["vehicle_type"] = fake_results["vehicle_type"]
        self.sim.results["response_time"] = fake_results["response_time"]

        for col in ["turnout_time", "on_scene_time", "travel_time"]:
            self.sim.results[col] = fake_results["times"]
        self.sim.results["priority"] = fake_results["priority"]
        self.sim.results["target"] = [600]*10

        ontime_score = self.sim.evaluate_performance(metric="on_time", vehicles=None, priorities=None, group_by=None, by_incident=True)
        meanresponsetime_score = self.sim.evaluate_performance(metric="mean_response_time", vehicles=None, priorities=None, group_by=None, by_incident=True)
        meanlateness_score = self.sim.evaluate_performance(metric="mean_lateness", vehicles=None, priorities=None, group_by=None, by_incident=True)
        assert ontime_score == approx(2/3), "On time score is off. Expected {}, got {}".format(2/3, ontime_score)
        assert meanresponsetime_score == approx((589 + 623 + 569)/3), "Mean response time is off"
        assert meanlateness_score == approx((0 + 23 + 0)/3), "Mean lateness is off"

        # with filter on priority
        ontime_score = self.sim.evaluate_performance(metric="on_time", vehicles=None, priorities=[1], group_by=None, by_incident=True)
        meanresponsetime_score = self.sim.evaluate_performance(metric="mean_response_time", vehicles=None, priorities=[1], group_by=None, by_incident=True)
        meanlateness_score = self.sim.evaluate_performance(metric="mean_lateness", vehicles=None, priorities=[1], group_by=None, by_incident=True)
        assert ontime_score == approx(1), "On time score is off. Expected {}, got {}".format(1, ontime_score)
        assert meanresponsetime_score == approx((589 + 569)/2), "Mean response time is off"
        assert meanlateness_score == approx(0), "Mean lateness is off"

        # with groupby priority
        ontime_scores = self.sim.evaluate_performance(metric="on_time", vehicles=None, priorities=None, group_by="priority", by_incident=True)
        meanresponsetime_scores = self.sim.evaluate_performance(metric="mean_response_time", vehicles=None, priorities=None, group_by="priority", by_incident=True)
        meanlateness_scores = self.sim.evaluate_performance(metric="mean_lateness", vehicles=None, priorities=None, group_by="priority", by_incident=True)
        assert list(ontime_scores.columns) == ["priority", "on_time"], "Column names not as expected"
        assert np.allclose(ontime_scores.sort_values("priority")["on_time"].values, np.array([1.0, 0.0])), "Values for on_time are off."
        assert np.allclose(meanresponsetime_scores.sort_values("priority")["mean_response_time"].values, np.array([(589+569)/2, 623])), "Mean response time is off"
        assert np.allclose(meanlateness_scores.sort_values("priority")["mean_lateness"].values, np.array([0.0, 23.0])), "Mean lateness is off"

# class TestDistributions(object):

#     @classmethod
#     def setup_class(self, data_dir=""):
#         self.deployments = pd.read_csv(".data/inzetten_2008-heden.csv", sep=";", decimal=",", low_memory=False)
#         self.incidents = pd.read_csv("./data/incidenten_2008-heden.csv", sep=";", decimal=",", low_memory=False)
#         self.station_locations = pd.read_excel("./data/kazernepositie en voertuigen.xlsx", sep=";", decimal=".")
#         self.vehicle_allocation = pd.read_csv("./data/voertuigenallocatie.csv", sep=";", decimal=".",
#                                               usecols=["kazerne", "#TS", "#RV", "#HV", "#WO"], nrows=19)
#         self.sim = Simulator(incidents, deployments, stations, vehicle_allocation)
#         self.sim.simulate_n_incidents(100000)

#     def test_incident_type_distributions(self):
#         pass

#     def test_

# if __name__=="__main__":
#     testr = TestSimulator()
#     testr.setup_class()
#     testr.test_create_vehicle_dict_normal()
