import os
import pandas as pd
import numpy as np
import itertools
import copy

from abc import abstractmethod, ABCMeta

import statsmodels.api as sm
import statsmodels.formula.api as smf

from math import gcd
from functools import reduce

from fdsim.helpers import progress, quick_load_simulator
import fdsim


class BaseExperiment():
    """Base class for experimentation with fdsim."""
    __metaclass__ = ABCMeta

    def __init__(self, forced_runs=None, max_runs=1000, effect_size=0.5, alpha=0.05, power=0.9,
                 name=None, description=None, verbose=True):
        self.name = name
        self.description = description
        self.forced_runs = forced_runs
        self.max_runs = max_runs
        self.effect_size = effect_size
        self.alpha = alpha
        self.power = power
        self.verbose = verbose

    @abstractmethod
    def run(self, n_runs=None):
        """Run the specified experiment."""

    @abstractmethod
    def analyze(self):
        """Analyze the simulation output by performing statistical tests."""

    @abstractmethod
    def print_results(self):
        """Print the results of the last experiment."""

    @abstractmethod
    def _determine_sample_size(self):
        """Determine the required number of observations per variant to ensure
        sufficient statistical power."""

    def set_evaluator(self, evaluator):
        assert isinstance(evaluator, fdsim.evaluation.Evaluator), "Expected an instance" \
            " of type fdsim.evaluation.Evaluator. Got {}.".format(type(evaluator))
        self.evaluator = evaluator

    @staticmethod
    def _reorganize_results_by_metric(results_dict):
        """Reorganize the data by measure and add scenario as column."""
        metric_data = {}
        for scenario, scenario_results in results_dict.items():
            for metric_set, results in scenario_results.items():
                results["scenario"] = scenario
                if metric_set in metric_data.keys():  # concat on previous data
                    metric_data[metric_set] = metric_data[metric_set].append(results, ignore_index=True)
                else:  # create new entry in dictionary
                    metric_data[metric_set] = results

        return metric_data

    def _lcm_two_numbers(self, x, y):
        """This function takes two integers and returns the Least Common Multiple."""
        lcm = (x * y) // gcd(x, y)
        return lcm

    def _lcm_n_numbers(self, *values):
        """Takes arbitrary number of integers and returns the Least Common Multiple."""
        return reduce(self._lcm_two_numbers, values)

    def _find_lcm_in_range(self, values, minimum, maximum):
        """Find the least common multiple (LCM) of an arbitrary number of integers
        within a given range."""
        lcm = self._lcm_n_numbers(*values)
        x = lcm
        while x <= maximum:
            if (x >= minimum) & (x <= maximum):
                return x
            x += lcm
        return None

    @staticmethod
    def _find_closest_higher_multiple(x, multiple, maximum=None):
        while True:
            if x % multiple == 0:
                break
            x += 1
        while x > maximum:
            x -= multiple
        return x

    def _get_final_n_runs(self, n, k_groups=2, find_lcm=True):
        """make sure number of runs does not exceed maximum and is an integer."""
        n = int(np.ceil(n))
        # calculate least common multiple if specified and multiple group sizes given
        if (not isinstance(k_groups, (int, float))) and (find_lcm is True):
            common_multiple = self._find_lcm_in_range(k_groups, n, self.max_runs)

            if common_multiple is not None:
                progress("Set number of runs to common multiple of group sizes: {}."
                         .format(common_multiple))
                return common_multiple

            else:
                final_n = int(np.minimum(n, self.max_runs))
                progress("No common multiple exists in the given range. Ignoring common "
                         "multiples. Using minimum of desired n and maximum: {}."
                         .format(final_n))

        elif (isinstance(k_groups, (int, float))) and (find_lcm is True):
            final_n = self._find_closest_higher_multiple(n, k_groups, maximum=self.max_runs)
            progress("Set number of runs to multiple of number of groups: {}."
                     " Number of groups: {}.".format(final_n, k_groups))

        # print what just happened
        elif n > self.max_runs:
            final_n = self.max_runs
            progress("Statistically desired number of runs larger than maximum ({} < {})."
                     " Using the maximum.".format(final_n, n))
        else:
            final_n = n
            progress("Desired total number of runs is {}.".format(final_n))

        return final_n

    @staticmethod
    def anova_table(aov):
        """Add effect sizes $\\eta^2 and \\omega^2$$ to an ANOVA table from statsmodels."""
        aov['mean_sq'] = aov[:]['sum_sq']/aov[:]['df']

        aov['eta_sq'] = aov[:-1]['sum_sq']/sum(aov['sum_sq'])

        aov['omega_sq'] = ((aov[:-1]['sum_sq'] - (aov[:-1]['df'] * aov['mean_sq'][-1])) / 
                           (sum(aov['sum_sq']) + aov['mean_sq'][-1]))

        cols = ['sum_sq', 'df', 'mean_sq', 'F', 'PR(>F)', 'eta_sq', 'omega_sq']
        aov = aov[cols]
        aov.columns = ['Sum of Squares', 'Degrees of Freedom', 'Mean Sum of Squares',
                       'F-statistic', 'P-value', 'Effect (eta^2)', 'Effect (omega^2)']
        return aov


class ABTest(BaseExperiment):
    """Class that performs A/B test to evaluate two scenarios against each other (usually
    one is the current situation (A) and one is a considered alternative (B)).
    """
    def __init__(self, sim_A=None, sim_B=None, evaluator=None, **kwargs):
        self.simulator_A = sim_A
        self.simulator_B = sim_B
        self.evaluator = evaluator
        self.ready = False
        self._check_readiness()

        super().__init__(**kwargs)

    def set_scenario_A(self, simulator, description=None):
        assert isinstance(simulator, fdsim.simulation.Simulator), \
            "Expected an instance of fdsim.simulation.Simulator as input."
        self.simulator_A = simulator
        self.description_A = description
        self._check_readiness()

    def set_scenario_B(self, simulator, description=None):
        assert isinstance(simulator, fdsim.simulation.Simulator), \
            "Expected an instance of fdsim.simulation.Simulator as input."
        self.simulator_B = simulator
        self.description_B = description
        self._check_readiness()

    def set_evaluator(self, evaluator):
        assert isinstance(evaluator, fdsim.evaluation.Evaluator), \
            "Expected an instance of fdsim.evaluation.Evaluator as input."
        self.evaluator = evaluator

    def run(self):
        if not self.ready:
            raise ValueError("Simulators A and B and an evaluator must be provided before"
                             " running.")

        self.n_runs = self.forced_runs or self._determine_sample_size()
        progress("Running scenario A for {} runs.".format(self.n_runs / 2))
        self.simulator_A.simulate_period(n=self.n_runs / 2)

        progress("Running scenario B for {} runs.".format(self.n_runs / 2))
        self.simulator_B.simulate_period(n=self.n_runs / 2)

        progress("Simulation completed. Evaluating results.")
        results_A = self.evaluator.evaluate(self.simulator_A.results)
        results_B = self.evaluator.evaluate(self.simulator_B.results)
        self.test_results = self.analyze(results_A, results_B)
        progress("Statistical tests performed and results obtained.")

    def analyze(self, results_A, results_B):
        """Analyze the output of the evaluator on the two simulation logs."""
        metric_set_names = list(results_A.keys())
        test_results = {}

        for set_name in metric_set_names:
            dfA = results_A[set_name]
            dfB = results_B[set_name]
            test_results[set_name] = {}

            for col in dfA.columns:
                tstat, pvalue, degrees = sm.stats.ttest_ind(dfA[col], dfB[col])
                estimate_A, estimate_B = dfA[col].mean(), dfB[col].mean()
                test_results[set_name][col] = {"estimate A": estimate_A,
                                               "estimate B": estimate_B,
                                               "t-statistic": tstat,
                                               "p-value": pvalue,
                                               "significant": pvalue < self.alpha / 2,
                                               "df": degrees}
        return test_results

    def print_results(self, to_latex=False):
        if self.test_results is None:
            progress("Nothing to print.")
        else:
            if self.name is None:
                additional = ""
            else:
                additional = ": " + self.name
            print("------------------\nA/B Test Results{}\n------------------"
                  .format(additional))
            if self.description is not None:
                print(self.description)

            if (self.description_A is not None) and (self.description_B is not None):
                print("Scenario A: {}".format(self.description_A))
                print("Scenario B: {}".format(self.description_B))
                print("------------------", end="\n\n")

            for name, f in self.test_results.items():
                print("Measure: '{}'\n{}"
                      .format(name, self.evaluator.metric_sets[name]["description"]))
                print("------------------------------------------------------------")
                if to_latex:
                    print(pd.DataFrame(f).T.to_latex())
                else:
                    print(pd.DataFrame(f).T)
                print("------------------------------------------------------------", end="\n\n")

    def _check_readiness(self):
        if ((self.simulator_A is not None) and (self.simulator_B is not None) and
                (self.evaluator is not None)):
            self.ready = True

    def _determine_sample_size(self):
        """Determine the required number of observations given the power, confidence, and
        effect size."""
        # solve equation for number of observations
        n = sm.stats.tt_ind_solve_power(
            effect_size=self.effect_size,
            alpha=self.alpha,
            power=self.power,
            ratio=1,
            alternative="two-sided"
        )
        # n*2 since it returns the sample size per variant
        return self._get_final_n_runs(n*2, find_lcm=True)


class MultiScenarioExperiment(BaseExperiment):

    def __init__(self, evaluator=None, **kwargs):
        super().__init__(**kwargs)
        self.evaluator = evaluator
        self.scenarios = {}
        self.descriptions = {}

    def add_scenario(self, simulator, name, description=None):
        """Add an alternative scenario to the possible options."""
        assert isinstance(name, str), "'name' must be a string. Got {}.".format(type(name))
        assert isinstance(simulator, fdsim.simulation.Simulator), ("'simulator' must be an "
            "instance of fdsim.simulation.Simulator. Got {}".format(type(simulator)))
        assert name not in self.scenarios.keys(), ("Scenario {} already exists"
            .format(name))

        self.scenarios[name] = simulator
        if description is not None:
            self.descriptions[name] = description

    def _determine_sample_size(self):
        """Determine the required number of observations given the power, confidence, and
        effect size."""
        # solve power equation for number of observations (nobs)
        total_n = sm.stats.FTestAnovaPower().solve_power(
            effect_size=self.effect_size,
            alpha=self.alpha,
            power=self.power,
            k_groups=len(self.scenarios),
            nobs=None
        )
        return self._get_final_n_runs(total_n, k_groups=len(self.scenarios), find_lcm=True)

    def run(self):
        # determine number of runs per scenario
        self.n_runs = self.forced_runs or self._determine_sample_size()
        self.scenario_runs = int(np.ceil(self.n_runs / len(self.scenarios)))
        # run scenarios
        results_dict = {}
        for scenario, sim in self.scenarios.items():
            progress("Running scenario {} for {} runs.".format(scenario, self.scenario_runs))
            sim.simulate_period(n=self.scenario_runs)
            progress("Computing performance metrics.")
            results_dict[scenario] = self.evaluator.evaluate(sim.results)
            progress("Simulation of scenario {} completed.".format(scenario))

        progress("Simulation completed. Conducting statistical analysis on results.")
        self.test_results = self.analyze(results_dict)
        progress("Statistical tests performed and results obtained.")

    def analyze(self, results_dict):
        """Analyze the results using one-way ANOVA."""
        # reorganize the data by measure and add scenario as column
        metric_data = self._reorganize_results_by_metric(results_dict)
        # save for post-hoc analysis
        self.metric_data = metric_data

        # analyze results per metric using one-way ANOVA
        anova_results = {}
        for measure, data in metric_data.items():
            tables = []
            for ycol in [c for c in data.columns if c != "scenario"]:
                # perform ANOVA
                ols_results = smf.ols("Q('{}') ~ C(scenario)".format(ycol), data=data).fit()
                table = self.anova_table(sm.stats.anova_lm(ols_results, typ=2))
                table.index.rename("predictor", inplace=True)
                table["target"] = ycol
                table = table.reset_index(drop=False)
                tables.append(table)

            measure_table = pd.concat(tables, axis=0)
            measure_table["predictor"].loc[measure_table["predictor"] == "C(scenario)"] = "scenario"
            measure_table.set_index(["target", "predictor"], inplace=True)
            measure_table["significant"] = measure_table["P-value"] < self.alpha / 2
            anova_results[measure] = measure_table

        significant_variable_dict = self.get_significant_results(anova_results)
        tukey_results = self.perform_tukey(metric_data, significant_variable_dict)

        self.anova_results = anova_results
        self.tukey_results = tukey_results
        return anova_results, tukey_results

    def get_significant_results(self, anova_results):
        """Find the dependent variables for which the scenario had a significant impact
        based on the ANOVA results."""
        target_dict = {}
        for measure, df in anova_results.items():
            df2 = df.reset_index()
            df2 = df2[(df2["significant"] == True) & (df2["predictor"] == "scenario")]
            targets = df2["target"].unique()
            target_dict[measure] = targets

        return target_dict

    def perform_tukey(self, data_dict, target_dict, group_col="scenario"):
        """Perform Tukey HSD post-hoc analysis to find the specific pairs of groups that had a
        significantly different mean."""
        outputs = {}
        for measure, data in data_dict.items():
            outputs[measure] = []
            for col in target_dict[measure]:
                tukey = sm.stats.multicomp.pairwise_tukeyhsd(data[col], data[group_col])
                columns = tukey.summary().data[0]
                summary = pd.DataFrame(tukey.summary().data[1:], columns=columns)
                summary.insert(0, column="metric", value=col)
                # add summary table to list
                outputs[measure].append(summary)

            if len(outputs[measure]) > 0:
                outputs[measure] = pd.concat(outputs[measure], axis=0, ignore_index=True)
            else:
                outputs[measure] = None

        return outputs

    def print_results(self, to_latex=False):
        """Print the results of the ANOVA and Tukey analysis to give a quick overview of the
        results."""
        if (self.anova_results is None) and (self.tukey_results is None):
            progress("Nothing to print.")
        else:
            if self.name is None:
                additional = ""
            else:
                additional = ": " + self.name
            print("------------------\nMultiple Scenarios Test Results{}\n------------------"
                  .format(additional))

            if len(self.descriptions) > 0:
                for name, description in self.descriptions.items():
                    print("Scenario '{}': {}".format(name, description))
                print("------------------", end="\n\n")

            for measure, aov_table in self.anova_results.items():
                print("\nMeasure: '{}'\n{}"
                      .format(measure, self.evaluator.metric_sets[measure]["description"]),
                      end="\n\n")
                print("Analysis of Variance (ANOVA)")
                print("------------------------------------------------------------")
                if to_latex:
                    print(aov_table.to_latex())
                else:
                    print(aov_table)
                print("------------------------------------------------------------", end="\n\n")

                print("Tukey HSD post-hoc analysis: pairwise comparison of relevant groups")
                print("------------------------------------------------------------")
                if to_latex:
                    print(self.tukey_results[measure].to_latex)
                else:
                    print(self.tukey_results[measure])
                print("------------------------------------------------------------", end="\n\n")


class MultiFactorExperiment(BaseExperiment):
    """Experiment to evaluate multiple scenarios against each other."""

    all_factor_types = ["resources", "location", "station_status", "add_station",
                        "remove_station", "activate_backup", "remove_backup"]

    def __init__(self, base_simulator, evaluator=None, cache_path="cache", **kwargs):

        self.evaluator = evaluator
        self.factors = {}
        self.factor_types = {}

        if not os.path.isdir(cache_path):
            os.mkdir(cache_path)
        self.cache_path = cache_path

        self.base_simulator = base_simulator
        self.base_simulator.save_simulator_object(os.path.join(cache_path, "base.fdsim"))

        super().__init__(**kwargs)

    def add_factor(self, factor_name, factor_type, levels=None):
        """Add a factor to the experiment. A factor is a variable that can take on at least
        two levels (categories).

        Parameters
        ----------
        factor_name: str,
            The name of the factor.
        factor_type: str,
            What type of factor it concerns. One of ["vehicles", "station_location",
            "station_status", "add_station", "remove_station", "crew"].
        levels: dict, optional (default: None),
            A nested dictionary specifying the parameters for the different levels, like:
            {'level_name_1' -> {'parameter 1' -> value1, 'parameter 2' -> value2, etc.},
             'level_name_2' -> {'parameter 1' -> value1, 'parameter 2' -> value2, etc.}}.
            If None, then it must be specified later by using self.add_factor_level().

        Notes
        -----
        The base_simulator's settings are also considered a level. I.e., make sure that the
        base case includes a relevant setting for each factor that is added.

        Examples
        --------
        >>> # add resources factor
        >>> experiment = MultiFactorExperiment(base_simulator)
        >>> levels = {"extra-ts": {"station_name": "HENDRIK", "TS": 1, "TS_crew_ft": 1}}
        >>> experiment.add_factor("TS-Hendrik", "resources", levels=levels)
        >>>
        >>> # add station_location factor with two alternatives to the base case
        >>> levels = {"at N200": {"new_location": "13781452"}}
                      "other-loc": {"new_location": "13781363"}}
        >>> experiment.add_factor("Osdorp loc", "location", "OSDORP", levels=levels)
        >>>
        >>> # set a station status cycle
        >>> levels = {"Closed at night":
                        {"station_name": "VICTOR", "start": 23, "end": 7, "status": "closed"},
                      "Office hours":
                        {"station_name": "VICTOR", "start": 18, "end": 8, "status": "closed"}
                     }
        >>> experiment.add_factor("Status Victor", "station_status", levels=levels)
        >>>
        >>> # add a station
        >>> levels = {"Bovenkerk":
                        {"station_name": "BOVENKERK", location": "13710002", "TS": 1, "TS_ft": 1},
                      "Harbor":
                        {"station_name": "HARBOR", location": "13710001", "TS": 1, "TS_ft": 1}
                     }
        >>> experiment.add_factor("New station location", "add_station", levels=levels)
        >>>
        >>> # remove a station
        >>> levels = {"Hendrik": {"station_name": "HENDRIK"},
                      "Nico": {"station_name": "NICO"}}
        >>> experiment.add_factor("Remove station", "remove_station", levels=levels)
        """
        if not factor_type in self.all_factor_types:
            raise ValueError("'{}' is not a valid factor type. Must be one of {}."
                .format(factor_type, self.all_factor_types))

        self.factor_types[factor_name] = factor_type
        self.factors[factor_name] = {"base": {}}
        if levels is not None:
            for i, (level_name, level_params) in enumerate(levels.items()):
                assert isinstance(level_params, dict), ("'levels' must be a nested dictionary"
                                                        " of at least 2 deep. Found a {} on "
                                                        "the second level."
                                                        .format(type(level_params)))

                self.factors[factor_name][level_name] = level_params

    def add_factor_level(self, factor_name, level_name, params):
        """Add a single level to an existing factor.

        Parameters
        ----------
        factor_name: str,
            The name of the factor to add a level to.
        level_name: str,
            The name of the new level.
        params: dict,
            The parameters to pass to the underlying method of fdsim.simulation.Simulator.
        """
        assert isinstance(factor_name, str), "'factor_name' must be a string."
        assert isinstance(level_name, str), "'factor_name' must be a string."
        assert level_name not in self.factors[factor_name].keys(), ("Level {} already exists"
            " in factor {}. Please choose another name.".format(level_name, factor_name))
        assert isinstance(params, dict), "'params' must be a dictionary."

        self.factors[factor_name][level_name] = params

    def run(self):
        """Run the experiment."""
        all_levels = [[level for level in factor.keys()] for factor in self.factors.values()]
        # determine sample size
        self.n_runs = self.forced_runs or self._determine_sample_size()
        self.n_scenarios = np.prod([len(levels) for levels in all_levels])
        self.scenario_runs = int(np.ceil(self.n_runs / self.n_scenarios))
        progress("Simulating {} runs in total ({} per scenario)."
                 .format(self.n_runs, self.scenario_runs))

        results_dict = {}
        scenario_dict = {}

        # run full factorial experiment design
        for k, level_combo in enumerate(itertools.product(*all_levels)):
            progress("Setting up combination {} / {}.".format(k + 1, self.n_scenarios))
            # load base simulator and apply level values
            simulator = quick_load_simulator(os.path.join(self.cache_path, "base.fdsim"))
            scenario = {}
            for i, level_name in enumerate(level_combo):
                factor_name = list(self.factors.keys())[i]
                scenario[factor_name] = level_name
                if level_name != "base":
                    simulator = self._apply_factor_level(simulator, factor_name, level_name)
                else:
                    # progress("Using base settings for factor {}.".format(factor_name))
                    pass

            # save experiment setup
            scenario_dict[k] = scenario
            progress("Factor levels: {}.".format(scenario))

            # run experiment
            progress("Running simulation for {} runs.".format(  self.scenario_runs))
            simulator.simulate_period(n=self.scenario_runs)
            progress("Computing performance metrics.")
            results_dict[k] = self.evaluator.evaluate(simulator.results)
            progress("Simulation of combination {} / {} completed.".format(k + 1, self.n_scenarios))

        # for debugging
        self.results_dict = results_dict
        self.scenario_dict = scenario_dict
        progress("results_dict and scenario_dict saved as attributes for debugging.")

        progress("Simulation completed. Conducting statistical analysis on results.")
        self.anova_results, self.tukey_results = self.analyze(results_dict, scenario_dict)

    def analyze(self, results_dict, scenario_dict):
        # reorganize the data by measure and add scenario as column
        metric_data = self._reorganize_results_by_metric(results_dict)

        # save for post-hoc analysis
        self.metric_data = metric_data

        progress("Adding used factor levels to results data.")
        # add factor information to data
        factor_names = list(self.factors.keys())
        for measure, data in metric_data.items():
            for fn in factor_names:
                data[fn] = data["scenario"].apply(lambda k: scenario_dict[k][fn])

        # create right hand side (rhs) for regression formula
        predictors = ["C(Q('{}'))".format(v) for v in factor_names]
        rhs = " + ".join(predictors)
        progress("Right hand side for OLS: {}".format(rhs))

        # analyze results per metric using multi-way ANOVA
        anova_results = {}
        for measure, data in metric_data.items():
            data.drop("scenario", axis=1, inplace=True)
            tables = []
            for ycol in [c for c in data.columns if c not in factor_names]:
                # perform ANOVA
                ols_results = smf.ols("Q('{}') ~ {}".format(ycol, rhs), data=data).fit()
                table = self.anova_table(sm.stats.anova_lm(ols_results, typ=2))
                table.index.rename("factor", inplace=True)
                table.insert(0, column="metric", value=ycol)
                table = table.reset_index(drop=False)
                tables.append(table)

            measure_table = pd.concat(tables, axis=0)
            measure_table["significant"] = measure_table["P-value"] < self.alpha / 2
            measure_table["factor"].loc[measure_table["factor"] != "Residual"] = \
                measure_table["factor"].loc[measure_table["factor"] != "Residual"].str[5:-3]
            anova_results[measure] = measure_table

        significant_variable_dict = self.get_significant_results(anova_results)
        tukey_results = self.perform_tukey(metric_data, significant_variable_dict)

        return anova_results, tukey_results

    def _apply_factor_level(self, simulator, factor_name, level_name):
        """Apply a factor level to a Simulator.

        Parameters
        ----------
        simulator: fdsim.simulation.Simulator,
            The simulator object to apply the settings to.
        factor_name: str,
            The name of the factor to change the level for.
        level_name: str,
            The name of the level to apply.

        Returns
        -------
        simulator: fdsim.simulation.Simulator,
            The simulator with adjusted settings.
        """
        level_params = copy.copy(self.factors[factor_name][level_name])
        station_name = level_params.pop("station_name")

        if self.factor_types[factor_name] == "resources":

            rs = simulator.resource_allocation.copy()
            rs.set_index("kazerne", inplace=True)
            cols = rs.columns

            for param, value in level_params.items():

                if param in cols:
                    rs.loc[station_name, param] = value

            rs = rs.astype(int).reset_index(drop=False)
            simulator.set_resource_allocation(rs)

        elif self.factor_types[factor_name] == "location":

            location = level_params.pop("new_location")
            simulator.move_station(station_name, location, **level_params)

        elif self.factor_types[factor_name] == "station_status":

            start = level_params.pop("start_hour")
            end = level_params.pop("end_hour")
            simulator.set_daily_station_status(station_name, start, end, **level_params)

        elif self.factor_types[factor_name] == "add_station":

            location = level_params.pop("location")
            simulator.add_station(station_name, location, **level_params)

        elif self.factor_types[factor_name] == "remove_station":

            simulator.remove_station(station_name)

        elif self.factor_types[factor_name] == "activate_backup":

            simulator.activate_backup_protocol(station_name, **level_params)

        elif self.factor_types[factor_name] == "remove_backup":

            simulator.remove_backup_protocol(station_name)

        return simulator

    def get_significant_results(self, anova_results):
        """Find the combinations of independent and dependent variables for which the scenario
        had a significant impact based on the ANOVA results."""
        sig_var_dict = {}
        for measure, df in anova_results.items():
            df2 = df.reset_index()  
            df2 = df2[(df2["significant"] == True) & (df2["factor"] != "Residual")]
            tuples = df2.apply(lambda x: (x["factor"], x["metric"]), axis=1)
            tuples = tuples.values.tolist()
            sig_var_dict[measure] = tuples

        return sig_var_dict

    def perform_tukey(self, data_dict, var_dict):
        """Perform Tukey HSD post-hoc analysis to find the specific pairs of groups that had a
        significantly different mean."""
        outputs = {}
        for measure, data in data_dict.items():
            outputs[measure] = []
            for predictor, target in var_dict[measure]:
                tukey = sm.stats.multicomp.pairwise_tukeyhsd(data[target], data[predictor])
                columns = tukey.summary().data[0]
                summary = pd.DataFrame(tukey.summary().data[1:], columns=columns)
                summary.insert(0, column="factor", value=predictor)
                summary.insert(0, column="metric", value=target)
                # add summary table to list
                outputs[measure].append(summary)

            if len(outputs[measure]) > 0:
                outputs[measure] = pd.concat(outputs[measure], axis=0, ignore_index=True)
            else:
                outputs[measure] = None

        return outputs

    def print_results(self, to_latex=False):
        """Print the results of the ANOVA and Tukey analysis to give a quick overview of the
        results."""
        if (self.anova_results is None) and (self.tukey_results is None):
            progress("Nothing to print.")
        else:
            if self.name is None:
                additional = ""
            else:
                additional = ": " + self.name
            print("------------------\nMultiple Factors Experiment Results{}\n------------------"
                  .format(additional))

            if len(self.factors) > 0:
                print("Factors:\n--------")
                for name, levels in self.factors.items():
                    print("Factor '{}' with levels: {}.".format(name, list(levels.keys())))
                print("------------------", end="\n\n")

            for measure, aov_table in self.anova_results.items():
                print("\nMeasure: '{}'\n{}"
                      .format(measure, self.evaluator.metric_sets[measure]["description"]),
                      end="\n\n")
                print("Analysis of Variance (ANOVA)")
                print("------------------------------------------------------------")
                if to_latex:
                    print(aov_table.to_latex())
                else:
                    print(aov_table)
                print("------------------------------------------------------------", end="\n\n")

                print("Tukey HSD post-hoc analysis: pairwise comparison of relevant groups")
                print("------------------------------------------------------------")
                if to_latex:
                    print(self.tukey_results[measure].to_latex)
                else:
                    print(self.tukey_results[measure])
                print("------------------------------------------------------------", end="\n\n")

    def _determine_sample_size(self):
        # solve equation for number of observations (nobs)
        group_sizes = [len(self.factors[f].keys()) for f in self.factors.keys()]
        n = sm.stats.FTestAnovaPower().solve_power(
            effect_size=self.effect_size,
            alpha=self.alpha,
            power=self.power,
            k_groups=np.max(group_sizes),
            nobs=None
        )
        progress("Statistically desired runs: {}".format(n))
        return self._get_final_n_runs(n, k_groups=group_sizes, find_lcm=True)
