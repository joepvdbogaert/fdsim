from pytest import approx
import numpy as np
import pandas as pd
import datetime

from fdsim.helpers import quick_load_simulator
import fdsim


class TestSimulator(object):

    @classmethod
    def setup_class(cls):
        cls.deployments = pd.read_csv("../Data/inzetten_2008-heden.csv", sep=";", decimal=",", low_memory=False)
        cls.incidents = pd.read_csv("../Data/incidenten_2008-heden.csv", sep=";", decimal=",", low_memory=False)
        cls.stations = pd.read_excel("../Data/kazernepositie en voertuigen.xlsx", sep=";", decimal=".")
        cls.resource_allocation = pd.read_csv("../Data/resource_allocation.csv", sep=";", decimal=".")
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

    def test_preprocess_resource_allocation(self):
        resource_allocation = self.sim._preprocess_resource_allocation(self.resource_allocation)
        assert np.all(resource_allocation["kazerne"] == resource_allocation["kazerne"].str.upper()), \
            "Station names are not upper cased."

    def test_create_vehicles_normal(self):
        vehicles = self.sim._create_vehicles(self.resource_allocation)
        assert isinstance(vehicles, dict), "Expected a dictionary"
        assert self.resource_allocation[["TS", "RV", "HV", "WO"]].sum().sum() == len(vehicles), "Number of vehicles does not match"
        assert len([v for v in vehicles.values() if v.type == "TS"]) == self.resource_allocation["TS"].sum(), "Incorrect number of TS vehicles"
        assert len([v for v in vehicles.values() if v.type == "RV"]) == self.resource_allocation["RV"].sum(), "Incorrect number of RV vehicles"
        assert len([v for v in vehicles.values() if v.type == "HV"]) == self.resource_allocation["HV"].sum(), "Incorrect number of HV vehicles"
        assert len([v for v in vehicles.values() if v.type == "WO"]) == self.resource_allocation["WO"].sum(), "Incorrect number of WO vehicles"

        for i in range(len(self.resource_allocation)):
            station_name = self.resource_allocation["kazerne"].iloc[i]
            assert len([v for v in vehicles.values() if v.base_station_name == station_name]) == self.resource_allocation.set_index("kazerne")[["TS", "RV", "HV", "WO"]].iloc[i].sum(), \
                "Incorrect number of vehicles for station {}".format(station_name)

    def test_create_stations(self):
        resources = self.sim._preprocess_resource_allocation(self.resource_allocation).copy()
        stations = self.sim._create_stations(resources)
        assert len(stations) == len(resources), "Incorrect number of stations created"
        assert np.all(np.in1d(resources["kazerne"].str.upper(), [s.name for s in stations.values()])), \
            "Not all station names created correctly."

        resources.set_index("kazerne", inplace=True)
        for station in stations.values():
            #for col in ["TS_crew_ft", "TS_crew_pt", "RVHV_crew_ft", "RVHV_crew_pt", "WO_crew_ft", "WO_crew_pt"]:
            assert np.all(station.crews["TS"] == [resources.loc[station.name, "TS_crew_ft"], resources.loc[station.name, "TS_crew_pt"]]), "TS crews wrong"
            assert np.all(station.crews["RVHV"] == [resources.loc[station.name, "RVHV_crew_ft"], resources.loc[station.name, "RVHV_crew_pt"]]), "RVHV crews wrong"
            assert np.all(station.crews["WO"] == [resources.loc[station.name, "WO_crew_ft"], resources.loc[station.name, "WO_crew_pt"]]), "WO crews wrong"

    def test_create_vehicles_empty(self):
        empty_allocation = self.resource_allocation.copy()
        for col in ["TS", "HV", "RV", "WO"]:
            empty_allocation[col] = 0
            print(empty_allocation)

        vehicles = self.sim._create_vehicles(empty_allocation)
        assert vehicles == dict(), "Expected emtpy dictionary"

    def test_create_vehicles_ts_only(self):
        allocation = self.resource_allocation.copy()
        for col in ["HV", "RV", "WO"]:
            allocation[col] = 0

        vehicles = self.sim._create_vehicles(allocation)
        assert np.all(np.unique([v.type for v in vehicles.values()]) == np.array(["TS"])), "Expected only TS vehicles"
        assert len([v for v in vehicles.values()]) == len([v for v in vehicles.values() if v.type == "TS"]), \
            "Expected all vehicles to be of type TS"

    def test_move_station_and_reset(self):
        station = "AMSTELVEEN"
        new_loc = "13710056"
        self.sim.move_station("AMSTELVEEN", "13710056", keep_name=True)
        assert np.all(self.sim._get_station_coordinates(station) == self.sim._get_coordinates(new_loc)), \
            "Coordinates of station not updated correctly"
        assert len(self.sim.rsampler.station_coords.keys()) == 19, "Length of station coordinates is off."

        station_index = np.nonzero(self.sim.dispatcher.station_names == station)[0][0]
        loc_index = np.nonzero(self.sim.dispatcher.matrix_names == new_loc)[0][0]
        assert (self.sim.dispatcher.time_matrix_stations[station_index, :] == self.sim.dispatcher.time_matrix[loc_index, :]).all(), \
            "Time matrix not updated correctly"
        for coords in [tuple(v.coords) for v in self.sim.vehicles.values() if v.current_station_name == station]:
            assert coords == self.sim._get_coordinates(new_loc), "Coordinate of vehicles not updated correctly"

        self.sim.reset_stations()
        assert (self.sim.dispatcher.time_matrix_stations[station_index, :] == self.sim.dispatcher.time_matrix_df.loc[station, :].values).all(), \
            "Time matrix not reset correctly."

    def test_set_vehicles(self):
        station_name = "OSDORP"
        vehicle_type = "TS"
        number = 5
        original_number = self.sim.resource_allocation.set_index("kazerne").loc[station_name, vehicle_type]
        assert self.sim.resource_allocation.set_index("kazerne").loc[station_name, vehicle_type] == original_number, \
            "Vehicle allocation not initialized correctly"
        assert len([v for v in self.sim.vehicles.values() if v.base_station_name == station_name and v.type == vehicle_type]) == original_number, \
            "Vehicle objects not initialized correctly"

        self.sim.set_vehicles(station_name, vehicle_type, number)
        assert self.sim.resource_allocation.set_index("kazerne").loc[station_name, vehicle_type] == number, \
            "Vehicle allocation not updated correctly"
        assert len([v for v in self.sim.vehicles.values() if v.base_station_name == station_name and v.type == vehicle_type]) == number, \
            "Vehicle objects not updated correctly"

        station_name_2 = "AMSTELVEEN"
        vehicle_type_2 = "WO"
        number_2 = 2
        self.sim.set_vehicles(station_name_2, vehicle_type_2, number_2)
        assert self.sim.resource_allocation.set_index("kazerne").loc[station_name_2, vehicle_type_2] == number_2, \
            "Vehicle allocation not updated correctly"
        assert len([v for v in self.sim.vehicles.values() if v.base_station_name == station_name_2 and v.type == vehicle_type_2]) == number_2, \
            "Vehicle objects not updated correctly"
        assert self.sim.resource_allocation.set_index("kazerne").loc[station_name, vehicle_type] == number, \
            "Vehicle allocation not updated correctly"
        assert len([v for v in self.sim.vehicles.values() if v.base_station_name == station_name and v.type == vehicle_type]) == number, \
            "Vehicle objects not updated correctly"

        self.sim.reset_stations()
        assert self.sim.resource_allocation.set_index("kazerne").loc[station_name, vehicle_type] == original_number, \
            "Vehicle allocation not initialized correctly"
        assert len([v for v in self.sim.vehicles.values() if v.base_station_name == station_name and v.type == vehicle_type]) == original_number, \
            "Vehicle objects not initialized correctly"
        self.sim.reset_stations()

    def test_relocate_vehicle(self):
        ids_amstelveen = [v.id for v in self.sim.vehicles.values() if v.type == "TS" and v.current_station_name == "AMSTELVEEN"]
        original_av = len(ids_amstelveen)
        ids_hendrik = [v.id for v in self.sim.vehicles.values() if v.type == "TS" and v.current_station_name == "HENDRIK"]
        original_hendrik = len(ids_hendrik)

        self.sim.relocate_vehicle("TS", "AMSTELVEEN", "HENDRIK")

        new_ids_av = [v.id for v in self.sim.vehicles.values() if v.type == "TS" and v.current_station_name == "AMSTELVEEN"]
        new_ids_hendrik = [v.id for v in self.sim.vehicles.values() if v.type == "TS" and v.current_station_name == "HENDRIK"]

        assert len(new_ids_av) == original_av - 1, "Number of vehicles at origin not updates correctly"
        assert len(new_ids_hendrik) == original_hendrik + 1, "Number of vehicles at destination not updated correctly"
        assert len(set(ids_amstelveen).union(set(ids_hendrik))) == len(set(new_ids_av).union(set(new_ids_hendrik))), "Total number of vehicles changed."
        assert set(ids_amstelveen).union(set(ids_hendrik)) == set(new_ids_av).union(set(new_ids_hendrik)), "Total set of IDs changed"
        assert set(ids_amstelveen).difference(set(new_ids_av)) == set(new_ids_hendrik).difference(set(ids_hendrik)), "ID of vehicle has gone missing"

        self.sim.relocate_vehicle("TS", "HENDRIK", "AMSTELVEEN")

    @staticmethod
    def see_activity(sim, station):
        times = sim.results[sim.results["station"] == station]["time"]
        times = pd.to_datetime(times)

        def count_per_hour(h):
            return np.sum(times.dt.hour == h)

        return [count_per_hour(h) for h in range(24)]

    @staticmethod
    def see_turnout_times_per_hour(sim, station, prio=1, vehicle="TS"):
        times = sim.results[(sim.results["station"] == station) &
                            (sim.results["priority"] == prio) &
                            (sim.results["vehicle_type"] == vehicle)][["time", "turnout_time"]]
        times["time"] = pd.to_datetime(times["time"])
        times["weekday"] = times["time"].dt.weekday

        def median_per_hour(d, h):
            return np.median(times[(times["time"].dt.hour == h) & (times["weekday"] == d)]["turnout_time"])

        medians = np.zeros((7, 24))
        for d in range(7):
            for h in range(24):
                medians[d, h] = median_per_hour(d, h)
        
        return medians

    def test_set_station_cycle_closed_at_night(self):

        self.sim.set_daily_station_status("VICTOR", start_hour=23, end_hour=7, status="closed")
        expected = np.array([[0., 0., 0., 0., 0., 0., 0., 2., 2., 2., 2., 2., 2., 2., 2., 2.,
                              2., 2., 2., 2., 2., 2., 2., 0.],
                             [0., 0., 0., 0., 0., 0., 0., 2., 2., 2., 2., 2., 2., 2., 2., 2.,
                              2., 2., 2., 2., 2., 2., 2., 0.],
                             [0., 0., 0., 0., 0., 0., 0., 2., 2., 2., 2., 2., 2., 2., 2., 2.,
                              2., 2., 2., 2., 2., 2., 2., 0.],
                             [0., 0., 0., 0., 0., 0., 0., 2., 2., 2., 2., 2., 2., 2., 2., 2.,
                              2., 2., 2., 2., 2., 2., 2., 0.],
                             [0., 0., 0., 0., 0., 0., 0., 2., 2., 2., 2., 2., 2., 2., 2., 2.,
                              2., 2., 2., 2., 2., 2., 2., 0.],
                             [0., 0., 0., 0., 0., 0., 0., 2., 2., 2., 2., 2., 2., 2., 2., 2.,
                              2., 2., 2., 2., 2., 2., 2., 0.],
                             [0., 0., 0., 0., 0., 0., 0., 2., 2., 2., 2., 2., 2., 2., 2., 2.,
                              2., 2., 2., 2., 2., 2., 2., 0.]])

        assert np.all(self.sim.stations["VICTOR"].status_table == expected), "Status table not as expected."

        # test if 'base_station' attribute of vehicles is updated as well
        vehicles_victor = [v for v in self.sim.vehicles.values() if v.base_station_name == "VICTOR"]
        for v in vehicles_victor:
            assert np.all(v.base_station.status_table == expected), "Status of vehicle's base station not updated correctly."

        # test simulation outputs
        self.sim.simulate_period(n=1)
        activity = self.see_activity(self.sim, "VICTOR")
        assert np.sum(activity[0:7]) == 0, "Activity spotted in closed period"
        assert np.sum(activity[23]) == 0, "Activity spotted in closed period"
        
    def test_set_station_cycle_parttime_in_weekend(self):

        self.sim.set_daily_station_status("HENDRIK", start_hour=23, end_hour=23, days_of_week=[5, 6], status="parttime")
        expected = np.array([[2.] * 24,
                             [2.] * 24,
                             [2.] * 24,
                             [2.] * 24,
                             [2.] * 24,
                             [1.] * 24,
                             [1.] * 24])

        assert np.all(self.sim.stations["HENDRIK"].status_table == expected), "Status table not as expected."

        # test if 'base_station' attribute of vehicles is updated as well
        vehicles_hendrik = [v for v in self.sim.vehicles.values() if v.base_station_name == "HENDRIK"]
        for v in vehicles_hendrik:
            assert np.all(v.base_station.status_table == expected), "Status of vehicle's base station not updated correctly."

        # test simulation outputs for adjusted turn-out times
        self.sim.simulate_period(n=2)
        turnout = self.see_turnout_times_per_hour(self.sim, "HENDRIK", prio=1, vehicle="TS")
        simulated_parttime = np.median([next(self.sim.rsampler.turnout_generators["parttime"][1]["TS"]) for i in range(10000)])
        simulated_fulltime = np.median([next(self.sim.rsampler.turnout_generators["fulltime"][1]["TS"]) for i in range(10000)])
        assert np.median(turnout[0:5, :]) == approx(simulated_fulltime, rel=0.1), "Median turnout times do not match"
        assert np.median(turnout[5:7, :]) == approx(simulated_parttime, rel=0.1), "Median turnout times do not match"

    def test_reset_all_station_status_cycles(self):
        self.sim.set_daily_station_status("VICTOR", start_hour=23, end_hour=7, status="closed")
        self.sim.set_daily_station_status("HENDRIK", start_hour=23, end_hour=23, days_of_week=[5, 6], status="parttime")
        self.sim.reset_all_station_status_cycles()
        assert np.all(self.sim.stations["VICTOR"].status_table == 2.), "Not all entries in status table are equal to 2."
        assert np.all(self.sim.stations["HENDRIK"].status_table == 2.), "Not all entries in status table are equal to 2."
        # test vehicle reference to stations
        vehicles_hendrik = [v for v in self.sim.vehicles.values() if v.base_station_name == "HENDRIK"]
        for v in vehicles_hendrik:
            assert np.all(v.base_station.status_table == 2.), "Status of vehicle's base station not updated correctly."

        vehicles_victor = [v for v in self.sim.vehicles.values() if v.base_station_name == "VICTOR"]
        for v in vehicles_victor:
            assert np.all(v.base_station.status_table == 2.), "Status of vehicle's base station not updated correctly."

    def test_set_start_time_end_time_and_reset_times(self):
        old_start_time = pd.to_datetime(self.sim.start_time)
        old_end_time = pd.to_datetime(self.sim.end_time)
        interval = (old_end_time - old_start_time).days
        # remove 1/4 on start and end
        delta = interval / 4

        new_start_time = old_start_time + pd.Timedelta(days=delta)
        # round down
        new_start_time = datetime.datetime(new_start_time.year, new_start_time.month, new_start_time.day, new_start_time.hour, 0)
        print("New start time: {}".format(new_start_time))
        self.sim.set_start_time(new_start_time)
        assert self.sim.start_time == new_start_time, "start_time attribute not updated correctly"
        assert self.sim.isampler.sampling_dict[0]["time"] == new_start_time, "start time of sampling dict not updated correctly"

        new_end_time = old_end_time - pd.Timedelta(days=delta)
        # round down
        new_end_time = datetime.datetime(new_end_time.year, new_end_time.month, new_end_time.day, new_end_time.hour, 0)
        print("New end time: {}".format(new_end_time))
        self.sim.set_end_time(new_end_time)
        assert self.sim.end_time == new_end_time, "end_time attribute not updated correctly"
        last_index = self.sim.isampler.T - 1
        assert self.sim.isampler.sampling_dict[last_index]["time"] == new_end_time, "end time of sampling dict not updated correctly"

        # reset
        self.sim.reset_simulation_period()
        assert self.sim.start_time == old_start_time, "Old start time not stored in attribute start_time"
        assert self.sim.end_time == old_end_time, "Old end time not stored in attribute end_time"
        assert self.sim.isampler.sampling_dict[0]["time"] == old_start_time, "start time of sampling dict not reset correctly"
        last_index = self.sim.isampler.T - 1
        assert self.sim.isampler.sampling_dict[last_index]["time"] == old_end_time, "end time of sampling dict not reset correctly"

    def test_log_result(self):
        n = 100
        self.sim.simulate_n_incidents(n, restart=True)
        assert isinstance(self.sim.results, pd.DataFrame)
        assert self.sim.results.shape[1] == 17, "Expected results dataframe of shape (100+, 17), got {}".format(self.sim.results.shape)
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
