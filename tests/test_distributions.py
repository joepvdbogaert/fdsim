import pytest
from pytest import approx
from fdsim.simulation import Simulator
from fdsim.incidentfitting import prepare_incidents_for_spatial_analysis
from fdsim.responsetimefitting import prepare_data_for_response_time_analysis
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
        # print("start loading sim")
        # cls.sim = Simulator(cls.incidents, cls.deployments, cls.stations, cls.vehicle_allocation)
        sim = quick_load_simulator()
        sim.simulate_n_incidents(500000)
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
        assert sim_dist_over_hours == approx(np.array([1.2, 1.2, 1.5, 1.5, 1.3])/np.sum([1.2, 1.2, 1.5, 1.5, 1.3]), abs=2e-2)

        sim_dist = simulated_incidents.groupby("incident_type")["t"].count().sort_index().values / len(simulated_incidents)
        A = np.array([[0.4, 0.6, 0.2, 0.5, 0.3], [0.1, 0.2, 0.5, 0.4, 0.2], [0.7, 0.4, 0.8, 0.6, 0.8]])
        expected_dist = A.dot(np.array([1.2, 1.2, 1.5, 1.5, 1.3])/np.sum([1.2, 1.2, 1.5, 1.5, 1.3]))
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
                simulated_dist.append(0)
            assert simulated_dist == approx(real_dist, abs=1e-2), "Priority distribution for type {} not as expected".format(type_)

    def test_spatial_distribution(self, sim, response_data):
        top_50_locations = np.array(response_data.drop_duplicates("dim_incident_id")
                                                 .groupby("hub_vak_bk")["dim_incident_id"]
                                                 .count()
                                                 .nlargest(50)
                                                 .index, dtype=str)

        top_50_simulated = np.array(list(self.sim_results.drop_duplicates("t")
                                                         .groupby("location")["t"]
                                                         .count()
                                                         .nlargest(50)
                                                         .index), dtype=str)

        assert np.sum(np.isin(top_50_simulated, top_50_locations)) >= 45, \
            "Of 50 most frequently sampled locations, less than 45 are also in the most frequent 50 in real data."

    def test_vehicle_distribution(self):
        deployments = self.deployments[np.isin(self.deployments["voertuig_groep"], ["TS", "RV", "HV", "WO"])].copy()
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

    def test_vehicles_per_incident(self):
        deployments = self.deployments[np.isin(self.deployments["voertuig_groep"], ["TS", "RV", "HV", "WO"])].copy()
        real = deployments.groupby("hub_incident_id")["voertuig_groep"].count().mean()
        simulated = self.sim_results.groupby("t")["vehicle_id"].count().mean()
        assert simulated == approx(real, abs=3e-2), "Mean number of vehicles per incident is not as in the data"

    @pytest.fixture
    def response_data(self):

        incidents = self.incidents.copy()
        deployments = self.deployments.copy()
        stations = self.stations.copy()

        # specify duplicate column names before merging
        incidents.rename({"kazerne_groep": "incident_kazerne_groep"},
                         axis="columns", inplace=True)
        deployments.rename({"kazerne_groep": "inzet_kazerne_groep"},
                           axis="columns", inplace=True)

        # merge incidents and deployments
        merged = pd.merge(deployments, incidents, how="inner",
                          left_on="hub_incident_id", right_on="dim_incident_id")

        # preprocess station name
        merged['inzet_kazerne_groep'] = merged['inzet_kazerne_groep'].str.upper()

        # rename x, y coordinate columns to avoid confusion
        merged.rename({"st_x": "incident_xcoord", "st_y": "incident_ycoord"},
                      axis="columns", inplace=True)
        # preprocess station name in same way as for the deployments
        stations["kazerne"] = stations["kazerne"].str.upper()

        # and rename to avoid confusion
        stations.rename({"lon": "station_longitude", "lat": "station_latitude"},
                        axis="columns", inplace=True)

        # create the merged dataset
        df = pd.merge(merged, stations, left_on="inzet_kazerne_groep",
                      right_on="kazerne", how="inner")

        # ensure data types
        df["inzet_rijtijd"] = df["inzet_rijtijd"].astype(float)

        # remove NaNs in location and cast to string
        df = df[~df["hub_vak_bk"].isnull()].copy()
        df["hub_vak_bk"] = df["hub_vak_bk"].astype(int).astype(str)
        df = df[df["hub_vak_bk"].str[0:2] == "13"]
        return df

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
        return np.mean(data), np.std(data)

    def test_dispatch_times_overall(self, dispatch_data):

        expected_mean, expected_std = self.calc_mean_and_std(dispatch_data["dispatch_time"].values)
        results = self.sim_results.drop_duplicates("t")
        simulated_mean, simulated_std = self.calc_mean_and_std(results["dispatch_time"].values)

        assert simulated_mean == approx(expected_mean, abs=5), "Overall mean dispatch time not as expected"
        assert simulated_std == approx(expected_std, abs=3), "Overall standard deviation of dispatch time not as expected"

    def test_dispatch_times_per_incident_type(self, dispatch_data):

        results = self.sim_results.drop_duplicates("t")

        for type_ in ["Binnenbrand", "Hulpverlening algemeen", "Reanimeren"]:
            # filter data
            type_data = dispatch_data[dispatch_data["dim_incident_incident_type"] == type_]
            simulated_type_data = results[results["incident_type"] == type_]
            # calculate and check dispatch time mean and std
            expected_mean, expected_std = self.calc_mean_and_std(type_data["dispatch_time"].values)
            simulated_mean, simulated_std = self.calc_mean_and_std(simulated_type_data["dispatch_time"].values)
            # check match
            assert simulated_mean == approx(expected_mean, abs=5), \
                "Mean dispatch time for {} not as expected".format(type_)
            assert simulated_std == approx(expected_std, abs=3), \
                "Standard deviation of dispatch time for {} not as expected".format(type_)

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

    def test_turnout_time_overall(self, turnout_data):
        expected_mean, expected_std = self.calc_mean_and_std(turnout_data["turnout_time"].values)
        simulated_mean, simulated_std = self.calc_mean_and_std(self.sim_results["turnout_time"].values)

        assert simulated_mean == approx(expected_mean, abs=5), "Overall mean turn-out time not as expected"
        assert simulated_std == approx(expected_std, abs=3), "Overall standard deviation of turn-out time not as expected"

    def test_turnout_times_per_incident_and_station(self, dispatch_data):

        for type_, station in [("Binnenbrand", "HENDRIK"), ("Hulpverlening algemeen", "AMSTELVEEN"), ("Reanimeren", "NICO")]:
            # filter data
            data = dispatch_data[(dispatch_data["dim_incident_incident_type"] == type_) &
                                 (dispatch_data["inzet_kazerne_groep"] == station)]

            simulated_data = self.sim_results[(self.sim_results["incident_type"] == type_) &
                                              (self.sim_results["station"] == station)]

            # calculate and check dispatch time mean and std
            expected_mean, expected_std = self.calc_mean_and_std(data["dispatch_time"].values)
            simulated_mean, simulated_std = self.calc_mean_and_std(simulated_data["dispatch_time"].values)

            assert simulated_mean == approx(expected_mean, abs=5), \
                "Mean turn-out time for {} not as expected".format(type_)
            assert simulated_std == approx(expected_std, abs=3), \
                "Standard deviation of turn-out time for {} not as expected".format(type_)

    def test_travel_time_distribution_overall(self):
        data = pd.read_csv("./data/response_data.csv")
        data = data[np.isin(data["voertuig_groep"], ["TS", "RV", "HV", "WO"])]
        data["speed"] = data["osrm_distance"] / data["inzet_rijtijd"] * 3.6
        data = data[(data["speed"] >= 16.5) & (data["speed"] <= 98.5)]

        expected_mean, expected_std = self.calc_mean_and_std(data["inzet_rijtijd"].values)
        simulated_mean, simulated_std = self.calc_mean_and_std(self.sim_results["travel_time"].values)

        assert simulated_mean == approx(expected_mean, abs=5), \
            "Overall mean travel time for {} not as expected"
        assert simulated_std == approx(expected_std, abs=5), \
            "Overall standard deviation of travel time not as expected"
