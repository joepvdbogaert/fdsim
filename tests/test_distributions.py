import pytest
from pytest import approx
from fdsim.simulation import Simulator
from fdsim.incidentfitting import prepare_incidents_for_spatial_analysis
from fdsim.responsetimefitting import prepare_data_for_response_time_analysis, sample_size_sufficient
from fdsim.helpers import quick_load_simulator
import pandas as pd
import numpy as np


class TestDistributions():

    @classmethod
    def setup_class(cls):
        cls.deployments = pd.read_csv("../Data/inzetten_2008-heden.csv", sep=";", decimal=",", low_memory=False)
        cls.incidents = pd.read_csv("../Data/incidenten_2008-heden.csv", sep=";", decimal=",", low_memory=False)
        cls.stations = pd.read_excel("../Data/kazernepositie en voertuigen.xlsx", sep=";", decimal=".")
        cls.vehicle_allocation = pd.read_csv("../Data/voertuigenallocatie.csv", sep=";", decimal=".",
                                             usecols=["kazerne", "#TS", "#RV", "#HV", "#WO"], nrows=19)
        cls.prepared_incidents = prepare_incidents_for_spatial_analysis(cls.incidents)
        cls.response_time_data = prepare_data_for_response_time_analysis(cls.incidents, cls.deployments,
                                                                         cls.stations, cls.vehicle_allocation)
        # print("start loading sim")
        # cls.sim = Simulator(cls.incidents, cls.deployments, cls.stations, cls.vehicle_allocation)
        sim = quick_load_simulator()
        sim.simulate_n_incidents(100000)
        cls.sim_results = sim.results.copy()
        # cls.sim.save_simulator_object()

    @classmethod
    def teardown_class(cls):
        del cls.sim

    @pytest.fixture
    def sim(self):
        return quick_load_simulator()

    def test_incident_type_distribution_and_distribution_over_time_intervals(self, sim):
        forecast = pd.DataFrame({"ds": ["15-11-2018 13:00:00", "15-11-2018 14:00:0", "15-11-2018 15:00:00",
                                        "15-11-2018 16:00:00", "15-11-2018 17:00:00"],
                                 "Binnenbrand": [0.4, 0.6, 0.2, 0.5, 0.3],
                                 "Buitenbrand": [0.1, 0.2, 0.5, 0.4, 0.2],
                                 "Reanimeren": [0.7, 0.4, 0.8, 0.6, 0.8]})
        sim.isampler.predictor.forecast = forecast
        sim.isampler.types = ["Binnenbrand", "Buitenbrand", "Reanimeren"]
        sim.isampler._set_sampling_dict(None, None, incident_types=None)
        sim.isampler._create_incident_types()
        sim.isampler._create_demand_locations()

        n = 100000
        sim.simulate_n_incidents(n)
        results = sim.results
        simulated_incidents = results.drop_duplicates("t")

        simulated_first_hour = simulated_incidents[pd.to_datetime(simulated_incidents["time"]) == pd.to_datetime("15-11-2018 13:00:00")]
        sim_dist_first_hour = simulated_first_hour.groupby("incident_type")["t"].count().sort_index().values / len(simulated_first_hour)
        assert sim_dist_first_hour == approx(np.array([0.4, 0.1, 0.7]) / 1.2, abs=2e-2),\
            "Distribution over incident type for first time interval not as expected"

        sim_dist_over_hours = simulated_incidents.groupby("time")["t"].count().sort_index().values / len(simulated_incidents)
        assert sim_dist_over_hours == approx(np.array([1.2, 1.2, 1.5, 1.5, 1.3])/np.sum([1.2, 1.2, 1.5, 1.5, 1.3]), abs=2e-2), \
            "Distribution over hours not as expected"

        sim_dist = simulated_incidents.groupby("incident_type")["t"].count().sort_index().values / len(simulated_incidents)
        A = np.array([[0.4, 0.6, 0.2, 0.5, 0.3], [0.1, 0.2, 0.5, 0.4, 0.2], [0.7, 0.4, 0.8, 0.6, 0.8]])
        expected_dist = A.dot(np.array([1.2, 1.2, 1.5, 1.5, 1.3])/np.sum([1.2, 1.2, 1.5, 1.5, 1.3]))
        expected_dist = expected_dist / np.sum(expected_dist)
        assert sim_dist == approx(expected_dist, abs=2e-2),\
            "Distribution over incident types not as expected"

    def test_priority_distribution(self, sim):
        incidents = self.incidents[~self.incidents["dim_prioriteit_prio"].isnull()]
        incidents = incidents[incidents["dim_prioriteit_prio"] != 5]
        assert np.all(np.isin(incidents["dim_prioriteit_prio"].unique(), [1, 2, 3])), \
            "Input data contains unexpected priority values"

        prio_dist = (incidents.groupby("dim_prioriteit_prio")["dim_incident_id"]
                              .count()
                              .sort_index()
                              .values / len(incidents))

        simulated_incidents = self.sim_results.drop_duplicates("t")
        simulated_dist = simulated_incidents.groupby("priority")["t"].count().sort_index().values / len(simulated_incidents)
        assert simulated_dist == approx(prio_dist, abs=2e-2)

    def test_priority_distribution_per_incident_type(self, sim):
        incidents = self.incidents[~self.incidents["dim_prioriteit_prio"].isnull()]
        incidents = incidents[incidents["dim_prioriteit_prio"] != 5]

        for type_ in ["Binnenbrand", "Buitenbrand", "Reanimeren"]:
            real_dist = (incidents[incidents["dim_incident_incident_type"] == type_]
                         .groupby("dim_prioriteit_prio")["dim_incident_id"]
                         .count()
                         .sort_index()
                         .values)

            real_dist = real_dist / np.sum(real_dist)

            simulated_dist = (self.sim_results[self.sim_results["incident_type"] == type_]
                              .drop_duplicates("t")
                              .groupby("priority")["t"]
                              .count()
                              .sort_index()
                              .values)

            simulated_dist = simulated_dist / np.sum(simulated_dist)

            if len(simulated_dist) == 2:
                simulated_dist = np.append(simulated_dist, 0)
            assert simulated_dist == approx(real_dist, abs=1e-2), "Priority distribution for type {} not as expected".format(type_)

    # def test_spatial_distribution(self, sim, response_data):
    #     response_data = response_data.drop_duplicates("dim_incident_id")
    #     results = self.sim_results.drop_duplicates("t").copy()
    #     types = self.get_types_with_more_observations(response_data, results, more_than=3000)

    #     for type_ in types:
    #         data = response_data[response_data["dim_incident_incident_type"] == type_]
    #         top_40_locations = np.array(data.drop_duplicates("dim_incident_id")
    #                                         .groupby("hub_vak_bk")["dim_incident_id"]
    #                                         .count()
    #                                         .nlargest(40)
    #                                         .index, dtype=str)

    #         top_40_simulated = np.array(list(results.drop_duplicates("t")
    #                                                 .groupby("location")["t"]
    #                                                 .count()
    #                                                 .nlargest(40)
    #                                                 .index), dtype=str)

    #         top_10_locations = np.array(data.drop_duplicates("dim_incident_id")
    #                                         .groupby("hub_vak_bk")["dim_incident_id"]
    #                                         .count()
    #                                         .nlargest(10)
    #                                         .index, dtype=str)

    #         top_10_simulated = np.array(list(results.drop_duplicates("t")
    #                                                 .groupby("location")["t"]
    #                                                 .count()
    #                                                 .nlargest(10)
    #                                                 .index), dtype=str)

    #         assert np.sum(np.isin(top_10_simulated, top_40_locations)) >= 10, \
    #             "Of 10 most frequently sampled locations for {}, less than 10 are also in the most frequent 30 in real data." \
    #             .format(type_)
    #         assert np.sum(np.isin(top_10_locations, top_40_simulated)) >= 10, \
    #             "Of 10 most frequent locations in the data for {}, less than 10 are also in the most frequently sampled 30." \
    #             .format(type_)

    def test_spatial_distribution_of_random_type_and_location(self, sim):
        """ Take a random type and location, sample a lot of locations for the type, and
        check whether the proportion is similar to that in the data. """
        for _ in range(5):
            # choose random incident type
            type_ = np.random.choice(self.prepared_incidents["dim_incident_incident_type"].unique())
            type_data = self.prepared_incidents[self.prepared_incidents["dim_incident_incident_type"] == type_]
            # get distribution in data over locations
            distribution_data = type_data.groupby("hub_vak_bk").size() / len(type_data)
            # sample from simulator.isampler and get sampled distribution
            sample = [sim.isampler.incident_types[type_].sample_location() for x in range(2000000)]
            # choose some random locations that has occurred for this type in the data
            for _ in range(20):
                # choose random location to test
                loc = np.random.choice(type_data["hub_vak_bk"].unique())
                proportion = sample.count(loc) / 1000000

                print("Testing proportion of {} that occurred in {}".format(type_, loc))                
                assert proportion == approx(distribution_data.loc[loc], rel=0.10), \
                    "Proportion of {} that happened in {} deviates".format(type_, loc)

    def test_vehicle_distribution(self):
        deployments = self.deployments[np.isin(self.deployments["voertuig_groep"], ["TS", "RV", "HV", "WO"])].copy()
        deployments = deployments[~deployments["voertuig_groep"].isnull()]
        real_dist = (deployments.groupby("voertuig_groep")["hub_inzet_id"]
                                .count()
                                .sort_index()
                                .values)

        real_dist = real_dist / np.sum(real_dist)

        simulated_dist = (self.sim_results.groupby("vehicle_type")["t"]
                                          .count()
                                          .sort_index()
                                          .values)
        simulated_dist = simulated_dist / np.sum(simulated_dist)

        assert simulated_dist == approx(real_dist, abs=1e-2), "Distribution over vehicle types not as expected."

    @staticmethod
    def get_types_with_more_observations(real, simulated, more_than=1500):
        """ Determine which incident types have more than 1000 observations
        in both the simulated as well as the real data. """
        sim_sizes = simulated.groupby("incident_type").size().sort_values(ascending=False)
        sim_big_types = list(sim_sizes[sim_sizes > more_than].index)
        real_sizes = real.groupby("dim_incident_incident_type").size().sort_values(ascending=False)
        real_big_types = list(real_sizes[real_sizes > more_than].index)
        intersection = set(sim_big_types).intersection(set(real_big_types))
        return list(intersection)

    def test_mean_vehicles_per_incident_for_each_incident_type(self):

        deployments = self.deployments[np.isin(self.deployments["voertuig_groep"], ["TS", "RV", "HV", "WO"])].copy()
        # add incident type to the deployment data
        deployments = deployments.merge(
            self.incidents[["dim_incident_id", "dim_incident_incident_type"]],
            left_on="hub_incident_id", right_on="dim_incident_id", how="inner")

        deployments = deployments[~deployments["voertuig_groep"].isnull()]

        vehicles = pd.DataFrame({})
        vehicles["simulated"] = self.sim_results.groupby(["t", "incident_type"]).size().groupby("incident_type").mean()
        vehicles["real"] = (deployments.groupby(["dim_incident_id", "dim_incident_incident_type"])
                                                             .size()
                                                             .groupby("dim_incident_incident_type")
                                                             .mean())

        big_types = self.get_types_with_more_observations(deployments.drop_duplicates("dim_incident_id"), self.sim_results.drop_duplicates("t"))
        
        vehicles = vehicles.loc[list(big_types)]
        
        assert vehicles["simulated"].values == approx(vehicles["real"].values, rel=0.05), \
            "Vehicles per incident not similar for all incident types. Types: {}".format(vehicles.index)

    # @pytest.fixture
    # def response_data(self):

    #     incidents = self.incidents.copy()
    #     deployments = self.deployments.copy()
    #     stations = self.stations.copy()

    #     # specify duplicate column names before merging
    #     incidents.rename({"kazerne_groep": "incident_kazerne_groep"},
    #                      axis="columns", inplace=True)
    #     deployments.rename({"kazerne_groep": "inzet_kazerne_groep"},
    #                        axis="columns", inplace=True)

    #     # merge incidents and deployments
    #     merged = pd.merge(deployments, incidents, how="inner",
    #                       left_on="hub_incident_id", right_on="dim_incident_id")

    #     # preprocess station name
    #     merged['inzet_kazerne_groep'] = merged['inzet_kazerne_groep'].str.upper()

    #     # rename x, y coordinate columns to avoid confusion
    #     merged.rename({"st_x": "incident_xcoord", "st_y": "incident_ycoord"},
    #                   axis="columns", inplace=True)
    #     # preprocess station name in same way as for the deployments
    #     stations["kazerne"] = stations["kazerne"].str.upper()

    #     # and rename to avoid confusion
    #     stations.rename({"lon": "station_longitude", "lat": "station_latitude"},
    #                     axis="columns", inplace=True)

    #     # create the merged dataset
    #     df = pd.merge(merged, stations, left_on="inzet_kazerne_groep",
    #                   right_on="kazerne", how="inner")

    #     # ensure data types
    #     df["inzet_rijtijd"] = df["inzet_rijtijd"].astype(float)

    #     # remove NaNs in location and cast to string
    #     df = df[~df["hub_vak_bk"].isnull()].copy()
    #     df["hub_vak_bk"] = df["hub_vak_bk"].astype(int).astype(str)
    #     df = df[df["hub_vak_bk"].str[0:2] == "13"]

    #     # only high priority incidents and deployments
    #     high_prio_data = df[(df["dim_prioriteit_prio"] == 1) &
    #                         (df["inzet_terplaatse_volgnummer"] == 1)].copy()
    #     return high_prio_data

    @pytest.fixture
    def response_data(self):

        df = self.response_time_data.copy()
        # only high priority incidents and deployments
        high_prio_data = df[(df["dim_prioriteit_prio"] == 1) &
                            (df["inzet_terplaatse_volgnummer"] == 1)].copy()

        return high_prio_data

    @pytest.fixture
    def dispatch_data(self, response_data):
        response_data["dispatch_time"] = (pd.to_datetime(response_data["inzet_gealarmeerd_datumtijd"], dayfirst=True) -
                                          pd.to_datetime(response_data["dim_incident_start_datumtijd"], dayfirst=True)).dt.seconds

        response_data = response_data[response_data["dispatch_time"] <= 600]
        response_data.drop_duplicates("dim_incident_id", inplace=True)
        return response_data

    @staticmethod
    def filter_extremes(data):
        n = len(data)
        return np.sort(data)[n//20:(n-n//20)]

    def calc_mean_and_std(self, data, filter=False):
        if filter:
            data = self.filter_extremes(data)
        data = data[~np.isnan(data)]
        return np.mean(data), np.std(data)

    # def test_dispatch_times_overall(self, dispatch_data):

    #     expected_mean, expected_std = self.calc_mean_and_std(dispatch_data["dispatch_time"].values)
    #     results = self.sim_results.drop_duplicates("t")
    #     simulated_mean, simulated_std = self.calc_mean_and_std(results["dispatch_time"].values)

    #     assert simulated_mean == approx(expected_mean, abs=5), "Overall mean dispatch time not as expected"
    #     assert simulated_std == approx(expected_std, abs=3), "Overall standard deviation of dispatch time not as expected"

    # def test_dispatch_times_per_incident_type(self, dispatch_data):

    #     results = self.sim_results.drop_duplicates("t")

    #     for type_ in results["incident_type"].unique():
    #         # filter data
    #         type_data = dispatch_data[dispatch_data["dim_incident_incident_type"] == type_]
    #         simulated_type_data = results[results["incident_type"] == type_]

    #         if (sample_size_sufficient(type_data["dispatch_time"])) and (type_ != "OMS / automatische melding"):
    #             # calculate and check dispatch time mean and std
    #             expected_mean, expected_std = self.calc_mean_and_std(type_data["dispatch_time"].values)
    #             simulated_mean, simulated_std = self.calc_mean_and_std(simulated_type_data["dispatch_time"].values)
    #             # check match
    #             assert simulated_mean == approx(expected_mean, abs=5), \
    #                 "Mean dispatch time for {} not as expected".format(type_)
    #             assert simulated_std == approx(expected_std, abs=15), \
    #                 "Standard deviation of dispatch time for {} not as expected".format(type_)
    #             print("Dispatch time test for {} executed".format(type_))
    #         else:
    #             print("Dispatch time test for {} cancelled due to insufficient observations".format(type_))

    def test_dispatch_times_per_incident_type(self, sim, dispatch_data):
        """ Compare dispatch times in data with those sampled from the IncidentSampler. """
        for type_ in dispatch_data["dim_incident_incident_type"].unique():

            # filter data
            type_data = dispatch_data[dispatch_data["dim_incident_incident_type"] == type_]

            if (sample_size_sufficient(type_data["dispatch_time"])) and (type_ != "OMS / automatische melding"):
                # calculate and check dispatch time mean and std
                expected_mean, expected_std = self.calc_mean_and_std(type_data["dispatch_time"].values)
                # sample
                sample = np.array([sim.rsampler.sample_dispatch_time(type_) for _ in range(10000)])
                simulated_mean, simulated_std = self.calc_mean_and_std(sample)
                # check match
                assert simulated_mean == approx(expected_mean, abs=5), \
                    "Mean dispatch time for {} not as expected".format(type_)
                assert simulated_std == approx(expected_std, abs=15), \
                    "Standard deviation of dispatch time for {} not as expected".format(type_)
                print("Dispatch time test for {} executed".format(type_))
            else:
                print("Dispatch time test for {} cancelled due to insufficient observations".format(type_))

    @pytest.fixture
    def turnout_data(self, response_data):
        # make station names uniform
        response_data["inzet_kazerne_groep"] = response_data["inzet_kazerne_groep"].str.upper()
        # calculate dispatch times
        response_data["turnout_time"] = (pd.to_datetime(response_data["inzet_uitgerukt_datumtijd"], dayfirst=True) -
                                         pd.to_datetime(response_data["inzet_gealarmeerd_datumtijd"], dayfirst=True)
                                         ).dt.seconds

        # filter out unrealistic values
        response_data = response_data[response_data["turnout_time"] <= 600]
        return response_data

    # def test_turnout_time_overall(self, turnout_data):
    #     print(turnout_data.head())
    #     expected_mean, expected_std = self.calc_mean_and_std(turnout_data["turnout_time"].values)
    #     print(self.sim_results["turnout_time"].head())
    #     print(len(self.sim_results["turnout_time"].isnull()))
    #     simulated_mean, simulated_std = self.calc_mean_and_std(self.sim_results["turnout_time"].values)

    #     assert simulated_mean == approx(expected_mean, abs=5), "Overall mean turn-out time not as expected"
    #     assert simulated_std == approx(expected_std, abs=3), "Overall standard deviation of turn-out time not as expected"

    def test_turnout_times_per_incident_and_station(self, turnout_data):

        for type_, station in [("Binnenbrand", "HENDRIK"), ("Hulpverlening algemeen", "AMSTELVEEN"), ("Reanimeren", "NICO")]:
            # filter data
            data = turnout_data[(turnout_data["dim_incident_incident_type"] == type_) &
                                (turnout_data["inzet_kazerne_groep"] == station)]

            if sample_size_sufficient(data["turnout_time"]):
                simulated_data = self.sim_results[(self.sim_results["incident_type"] == type_) &
                                                  (self.sim_results["station"] == station)]

                # calculate and check dispatch time mean and std
                expected_mean, expected_std = self.calc_mean_and_std(data["turnout_time"].values)
                simulated_mean, simulated_std = self.calc_mean_and_std(simulated_data["turnout_time"].values)

                assert simulated_mean == approx(expected_mean, abs=5), \
                    "Mean turn-out time for {} at {} not as expected".format(type_, station)
                assert simulated_std == approx(expected_std, abs=10), \
                    "Standard deviation of turn-out time for {} at {} not as expected".format(type_, station)
                print("Turnout time test executed for {} {}".format(type_, station))
            else:
                print("Type {}, station {} did not have sufficient observations.".format(type_, station))

    # def test_travel_time_distribution_overall(self):
    #     data = pd.read_csv("./data/response_data.csv", dtype={"hub_vak_bk": int}, low_memory=False)
    #     data = data[np.isin(data["voertuig_groep"], ["TS", "RV", "HV", "WO"])]
    #     data = data[data["dim_prioriteit_prio"] == 1]
    #     data = data[~data["inzet_rijtijd"].isnull()]
    #     data["speed"] = data["osrm_distance"] / data["inzet_rijtijd"] * 3.6
    #     data = data[(data["speed"] >= 16.5) & (data["speed"] <= 98.5)]

    #     expected_mean, expected_std = self.calc_mean_and_std(data["inzet_rijtijd"].values)
    #     simulated_mean, simulated_std = self.calc_mean_and_std(self.sim_results["travel_time"].values)

    #     assert simulated_mean == approx(expected_mean, abs=5), \
    #         "Overall mean travel time not as expected"
    #     assert simulated_std == approx(expected_std, abs=5), \
    #         "Overall standard deviation of travel time not as expected"

    def test_travel_times_for_specific_routes(self, sim):
        """ Take some routes that occur frequently and see if distributions are similar. """
        data = pd.read_csv("./data/response_data.csv", dtype={"hub_vak_bk": int}, low_memory=False)
        data["hub_vak_bk"] = data["hub_vak_bk"].astype(str)
        data = data[data["voertuig_groep"] == "TS"]
        data = data[data["dim_prioriteit_prio"] == 1]
        data = data[~data["inzet_rijtijd"].isnull()]
        data["speed"] = data["osrm_distance"] / data["inzet_rijtijd"] * 3.6
        data = data[(data["speed"] >= 16.5) & (data["speed"] <= 98.5)]

        biggest = data.groupby(["inzet_kazerne_groep", "hub_vak_bk"]).size().nlargest(20)
        routes = biggest.index.tolist()
        for route in routes:
            print("Testing route: {}".format(route))
            real_tt_mean = data[(data["inzet_kazerne_groep"] == route[0]) & (data["hub_vak_bk"] == route[1])]["inzet_rijtijd"].mean()
            real_tt_std = data[(data["inzet_kazerne_groep"] == route[0]) & (data["hub_vak_bk"] == route[1])]["inzet_rijtijd"].std()
            # sample
            # note that type does not influence travel time, so we input a random value
            expected_time = sim.dispatcher.time_matrix_df.loc[route[0], route[1]]
            sample = [sim.rsampler.sample_travel_time(expected_time, "TS") for _ in range(10000)]
            sim_tt_mean = np.mean(sample)
            sim_tt_std = np.std(sample)
            assert sim_tt_mean == approx(real_tt_mean, abs=15), "Mean travel time from {} to {}".format(route[0], route[1])
            assert sim_tt_std == approx(real_tt_std, abs=15), "Std of travel time from {} to {}".format(route[0], route[1])

    @pytest.fixture
    def onscene_data(self, response_data):
        # calculate dispatch times
        response_data["on_scene_time"] = (pd.to_datetime(response_data["inzet_eind_inzet_datumtijd"], dayfirst=True) -
                                         pd.to_datetime(response_data["inzet_start_inzet_datumtijd"], dayfirst=True)
                                         ).dt.seconds

        # filter out unrealistically small and large values
        response_data = response_data[(response_data["on_scene_time"] > 60) &
                                      (response_data["on_scene_time"] < 60*60*24)].copy()

        return response_data

    def test_onscene_times_priority_1(self, sim, onscene_data):
        """ Test the on-scene times of different incident and vehicle types. """
        onscene_data = onscene_data[onscene_data["dim_prioriteit_prio"] == 1]

        for incident_type in onscene_data["dim_incident_incident_type"].unique():
            incident_type_data = onscene_data[onscene_data["dim_incident_incident_type"] == incident_type]

            for vehicle_type in incident_type_data["voertuig_groep"].unique():
                # sample
                sample = np.array([next(sim.rsampler.onscene_generators[incident_type][vehicle_type]) for _ in range(10000)])
                real_mean = incident_type_data[incident_type_data["voertuig_groep"] == vehicle_type]["on_scene_time"].mean()
                real_std = incident_type_data[incident_type_data["voertuig_groep"] == vehicle_type]["on_scene_time"].std()
                assert np.mean(sample) == approx(real_mean, abs=60), "Mean on-scene time for {}, {}".format(incident_type, vehicle_type)
                assert np.std(sample) == approx(real_std, abs=60), "Std of on-scene time for {}, {}".format(incident_type, vehicle_type)
