"""
:code:`fdsim` provides an easy and flexible experimentation interface that takes care of
using the appropriate statistical tests and number of simulation runs, so that you don't
have to.

There are three usable experiment setups available: :code:`ABTest` to compare two alternative
scenarios, :code:`MultiScenarioExperiment` to compare more than two alternatives, and the
:code:`MultiFactorExperiment` for experiments with multiple independent decision variables.

All you have to do is choose the right experiment for your question, set up the corresponding
experiment class and run it.

Setting up the experiments differs somewhat between the three cases, because they are of
different complexity.
"""
import os
import pandas as pd
import numpy as np
import itertools
import copy
import pickle

from abc import abstractmethod, ABCMeta

import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

from math import gcd
from functools import reduce

from fdsim.helpers import progress, quick_load_simulator
import fdsim


class BaseExperiment():
    """Base class for experimentation with fdsim. Not useful to instantiate on its own.

    Parameters
    ----------
    forced_runs: int or None, optional, default=None
        A fixed number of simulation runs to use for the entire experiment (so this number
        will be divided over the number of scenarios.
    max_runs: int, optional, default=1000
        The maximum number of runs to use in the experiment.
    effect_size: float, optional, default=0.5
        The effect size, normalized by the standard deviation, that you want to be able to
        detect in the experiment. The lower the value, the more runs will be used (up to
        :code:`max_runs` and provided that :code:`forced_runs=None`).
    alpha: float, optional, default=0.05
        The allowed probability of falsely discovering a significant effect. One minus the
        significance level. Using :math:`\\alpha=0.05` is conventional in statistical
        experiments.
    power: float, optional, default=0.8
        The statistical power, i.e., the probability of finding a significant effect if it
        exists. One minus the power is the probability of a Type II error (false negatives).
    name: str, optional, default=None
        The name of the experiment. Will be used when printing results.
    description: str, optional, default=None
        Description of the experiment. Will be used when printing results.
    to_minutes: bool, optional, default=True
        Whether to translate the results to minutes (mm:ss) or to leave it in full seconds.
    with_std: bool, optional, default=True
        Whether to include the standard deviation of a metric among several runs with the same
        settings in the results tables.
    pvalue_zero_threshold: float, optional, default=1e-8
        From which value to print p-values as zeros in results tables.
    verbose: bool, optional, default=True
        Whether to print progress during experiment runs.
    """
    __metaclass__ = ABCMeta

    not_to_minutes = ["count", "prop_missing"]

    def __init__(self, forced_runs=None, max_runs=1000, effect_size=0.5, alpha=0.05, power=0.8,
                 name=None, description=None, to_minutes=True, with_std=True, pvalue_zero_threshold=1e-8, verbose=True):
        self.name = name
        self.description = description
        self.forced_runs = forced_runs
        self.max_runs = max_runs
        self.effect_size = effect_size
        self.alpha = alpha
        self.power = power
        self.to_minutes = to_minutes
        self.with_std = with_std
        self.pvalue_zero_threshold = pvalue_zero_threshold
        self.verbose = verbose

    @abstractmethod
    def run(self, n_runs=None):
        """Run the specified experiment."""

    @abstractmethod
    def analyze(self):
        """Analyze the simulation output by performing statistical tests."""

    @abstractmethod
    def save_simulation_logs(self, dirpath=".", prefix="experiment1"):
        """Save the raw simulation logs to disk."""

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

    def _solve_power(self, k):
        n = sm.stats.FTestAnovaPower().solve_power(
            effect_size=self.effect_size,
            alpha=self.alpha,
            power=self.power,
            k_groups=k,
            nobs=None
        )
        return n

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

    Parameters
    ----------
    sim_A, sim_B: fdsim.simulation.Simulator, optional, default=None
        The simulators for scenario A and B. If None, these must be provided later, using
        the :code:`set_scenario_A` and :code:`set_scenario_B` methods.
    evaluator: fdsim.evaluation.Evaluator, optional, default=None
        The :code:`Evaluator` object that should be used to analyze the simulation
        runs. This class specifies which metrics will be compared between the two
        scenarios.
    **kwargs: key-value pairs
        Any parameters that should be passed to the BaseSimulator (see :code:`BaseExperiment`)
        for the available parameters.
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
        self.test_results = self.analyze(results_A, results_B, to_minutes=self.to_minutes,
                                         with_std=self.with_std)
        progress("Statistical tests performed and results obtained.")

    def analyze(self, results_A, results_B, to_minutes=True, with_std=True):
        """Analyze the output of the evaluator on the two simulation logs."""
        def seconds_to_minutes(*args):
            return ["{:02.0f}:{:02.0f}".format(x // 60, x % 60) for x in args]

        def round(value, measure, descriptor):
            if (measure == "on_time") and (descriptor != "count"):
                return np.round(value, 4)
            elif descriptor == "prop_missing":
                return np.round(value, 5)
            else:
                return int(np.round(value))

        metric_set_names = list(results_A.keys())
        test_results = {}

        for set_name in metric_set_names:
            dfA = results_A[set_name]
            dfB = results_B[set_name]
            test_results[set_name] = {}
            set_measure = self.evaluator.metric_set_measures[set_name]
            if set_measure == "on_time":
                digits = 4
            else:
                digits = 0

            for col in dfA.columns:
                tstat, pvalue, degrees = sm.stats.ttest_ind(dfA[col], dfB[col])
                mean_A, mean_B = round(dfA[col].mean(), set_measure, col), round(dfB[col].mean(), set_measure, col)
                # add standard deviation to the estimates
                if with_std:
                    std_A, std_B = round(dfA[col].std(), set_measure, col), round(dfB[col].std(), set_measure, col)
                    if to_minutes and (col not in self.not_to_minutes) and (set_measure is not "on_time"):
                        estimate_A, estimate_B, std_A, std_B, effect = \
                            seconds_to_minutes(mean_A, mean_B, std_A, std_B, (mean_B - mean_A))
                    else:
                        estimate_A, estimate_B = mean_A, mean_B
                        effect = estimate_B - estimate_A

                    # format as "mean (std)"
                    estimate_A = "{} ({})".format(estimate_A or mean_A, std_A)
                    estimate_B = "{} ({})".format(estimate_B or mean_B, std_B)
                # no standard deviation, but do change to minutes:seconds formatting
                elif to_minutes and (col not in self.not_to_minutes) and (set_measure is not "on_time"):
                    estimate_A, estimate_B, effect = seconds_to_minutes(mean_A, mean_B, (mean_B - mean_A))
                # use mean in seconds only
                else:
                    estimate_A, estimate_B, effect = mean_A, mean_B, mean_B - mean_A
                test_results[set_name][col] = {"A estimate": estimate_A,
                                               "B estimate": estimate_B,
                                               "effect": effect,
                                               "t-statistic": tstat,
                                               "p-value": pvalue if pvalue > self.pvalue_zero_threshold else 0,
                                               "significant": pvalue < self.alpha / 2,
                                               "df": degrees}
        return test_results

    def save_simulation_logs(self, dirpath=".", prefix="experiment1"):
        """Write simulation logs to disk.

        Parameters
        ----------
        dirpath: str
            Path to directory to save the result in.
        prefix: str
            String to prepend to '_results_A' and 'results_B' for the two logs.
        """
        self.simulator_A.results.to_csv(os.path.join(dirpath, prefix + "_results_A.csv"))
        self.simulator_B.results.to_csv(os.path.join(dirpath, prefix + "_results_B.csv"))

    def print_results(self, to_latex=False, plot=True):
        """Print the results of the experiments to stdout.

        Parameters
        ----------
        to_latex: bool, optional, default=True
            Whether to print tables as LaTeX Tabular environments.
        plot: bool, optional, default=True
            Whether to print distribution plots as well.
        """
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
                print("------------------", end="\n")

            print("Runs: {} ({} per scenario)".format(self.n_runs, self.n_runs / 2))
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
                if plot and (self.evaluator.metric_set_measures[name] in ["response_time", "delay"]):
                    fig = self.evaluator.plot(name, self.simulator_A.results, self.simulator_B.results,
                                              labels=["Scenario A", "Scenario B"], return_fig=True)
                    plt.plot()

    def plot_distributions(self, metric_set_name, return_fig=True):
        """Plot figures showing the probability of exceeding any given value for a measure.

        Parameters
        ----------
        metric_set_name: str
            The name of the metric set in the :code:`Evaluator` object that you want to plot.
        return_fig: bool, optional, default=True
            Whether to return the figure or just print it.
        """
        fig = self.evaluator.plot(metric_set_name, self.simulator_A.results, self.simulator_B.results,
                                  labels=["Scenario A", "Scenario B"], return_fig=True)
        if return_fig:
            return fig
        else:
            plt.show()

    def _check_readiness(self):
        if ((self.simulator_A is not None) and (self.simulator_B is not None) and
                (self.evaluator is not None)):
            self.ready = True

    def _determine_sample_size(self):
        """Determine the required number of observations given the power, confidence, and
        effect size."""
        n = self._solve_power(2)
        return self._get_final_n_runs(n*2, find_lcm=True)


class MultiScenarioExperiment(BaseExperiment):
    """Class that is used to evaluate more than two alternative scenarios against
    each other.

    Scenarios can be added one by one using the :code:`add_scenario` method. Since this method
    takes initialized and configured :code:`fdsim.simulation.Simulator`s as input, no
    assumptions are made on the setups of the scenarios. Each is treated as an alternative
    situation that is compared against all other situations/scenarios.

    Parameters
    ----------
    evaluator: fdsim.evaluation.Evaluator, optional, default=None
        The :code:`Evaluator` object that should be used to analyze the simulation
        runs. This class specifies which metrics will be compared between the two
        scenarios.
    **kwargs: key-value pairs
        Any parameters that should be passed to the BaseSimulator (see :code:`BaseExperiment`)
        for the available parameters.
    """
    def __init__(self, evaluator=None, **kwargs):
        super().__init__(**kwargs)
        self.evaluator = evaluator
        self.scenarios = {}
        self.descriptions = {}

    def add_scenario(self, simulator, name, description=None):
        """Add an alternative scenario to the possible options.

        Parameters
        ----------
        simulator: fdsim.simulation.Simulator
            The simulator that is configured to simulate the desired scenario.
        name: str
            The name for this scenario. It is used to print and distinguish between
            scenarios later on and is therefore required.
        description: str, optional, default=None
            An optional description of this scenario that may provide additional context
            to the user.
        """
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
        total_n = self._solve_power(len(self.scenarios))
        return self._get_final_n_runs(total_n, k_groups=len(self.scenarios), find_lcm=True)

    def run(self):
        """Run the experiment."""
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
        """Analyze the results using one-way ANOVA.

        Parameters
        ----------
        results_dict: dict
            Dictionary where keys are scenario names and values are outputs of
            :code:`Evaluator.evaluate`.
        """
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

    def save_simulation_logs(self, dirpath=".", prefix="experiment1"):
        """Write simulation logs to disk.

        Parameters
        ----------
        dirpath: str
            Path to directory to save the result in.
        prefix: str
            String to prepend to '_results_A' and 'results_B' for the two logs.
        """
        for name, sim in self.scenarios.items():
            sim.results.to_csv(os.path.join(dirpath, prefix + "_scenario_{}.csv".format(name)))

    def print_results(self, to_latex=False, plot=True):
        """Print the results of the ANOVA and Tukey analysis to give a quick overview of the
        results.

        Parameters
        ----------
        to_latex: bool, optional, default=True
            Whether to print tables as LaTeX Tabular environments.
        plot: bool, optional, default=True
            Whether to print distribution plots as well.
        """
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

            print("Runs: {} ({} per scenario)".format(self.n_runs, self.scenario_runs))
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

                if plot and (self.evaluator.metric_set_measures[measure] in ["response_time", "delay"]):
                    print("Probabilities of exceeding various response times")
                    self.evaluator.plot(measure, *[sim.results for sim in self.scenarios.values()],\
                                        labels=[scenario for scenario in self.scenarios.keys()])

    def plot_distributions(self, metric_set_name, return_fig=True):
        """Plot figures showing the probability of exceeding any given value for a measure.

        Parameters
        ----------
        metric_set_name: str
            The name of the metric set in the :code:`Evaluator` object that you want to plot.
        return_fig: bool, optional, default=True
            Whether to return the figure or just print it.
        """
        fig = self.evaluator.plot(metric_set_name, *[sim.results for sim in self.scenarios.values()],
                                  labels=list(self.scenarios.keys()), return_fig=True)
        if return_fig:
            return fig
        else:
            plt.show()


class MultiFactorExperiment(BaseExperiment):
    """Experiment to evaluate combinations of values for multiple decision variables / factors.

    This has a slightly more complex API due to the fact that different factor levels should
    be applied to a :code:`Simulator` in order to define every possible setting. How this works
    is explained in the :code:`add_factor` method.

    Parameters
    ----------
    base_simulator: fdsim.simulation.Simulator
        The base situation to which different factor levels will be applied. Note that for
        every factor in the experiment, a relevant value should be set in this base situation.
    evaluator: fdsim.evaluation.Evaluator
        The :code:`Evaluator` object to use to calculate performance measures and metrics.
    cache_path: str
        Directory to store base simulator in, so that it can be reloaded for different runs.
    **kwargs: key-value pairs
        Parameters passed to :code:`BaseExperiment`.
    """

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

        Change the location of stations:

        .. code::

            >>> # add station_location factor with two alternatives to the base case
            >>> levels = {"at N200": {"new_location": "13781452"}}
                          "other-loc": {"new_location": "13781363"}}
            >>> experiment.add_factor("Osdorp loc", "location", "OSDORP", levels=levels)

        Set a `status cycle` such as closing a station part of the day or having part time
        crew operate it:

        .. code:: python

            >>> # set a station status cycle
            >>> levels = {"Closed at night":
                            {"station_name": "VICTOR", "start": 23, "end": 7, "status": "closed"},
                          "Office hours":
                            {"station_name": "VICTOR", "start": 18, "end": 8, "status": "closed"}
                         }
            >>> experiment.add_factor("Status Victor", "station_status", levels=levels)

        .. code:: python

            >>> # add a station
            >>> levels = {"Bovenkerk":
                            {"station_name": "BOVENKERK", location": "13710002", "TS": 1, "TS_ft": 1},
                          "Harbor":
                            {"station_name": "HARBOR", location": "13710001", "TS": 1, "TS_ft": 1}
                         }
            >>> experiment.add_factor("New station location", "add_station", levels=levels)

        .. code:: python

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
        self.scenario_logs = {}

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
            self.scenario_logs[k] = simulator.results.copy()
            progress("Computing performance metrics.")
            results_dict[k] = self.evaluator.evaluate(simulator.results)
            progress("Simulation of combination {} / {} completed.".format(k + 1, self.n_scenarios))

        # for debugging
        self.results_dict = results_dict
        self.scenario_dict = scenario_dict
        # progress("results_dict and scenario_dict saved as attributes for debugging.")

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

    def print_results(self, to_latex=False, plot=True):
        """Print the results of the ANOVA and Tukey analysis to give a quick overview of the
        results.

        Parameters
        ----------
        to_latex: bool, optional, default=True
            Whether to print tables as LaTeX Tabular environments.
        plot: bool, optional, default=True
            Whether to print distribution plots as well.
        """
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

                if plot and (self.evaluator.metric_set_measures[measure] in ["response_time", "delay"]):
                    print("Probabilities of exceeding various response times")
                    self.evaluator.plot(measure, *[log for log in self.scenario_logs.values()],
                                        labels=["Scenario {}".format(k) for k in self.scenario_logs.keys()])

    def plot_distributions(self, metric_set_name, return_fig=True):
        """Plot figures showing the probability of exceeding any given value for a measure.

        Parameters
        ----------
        metric_set_name: str
            The name of the metric set in the :code:`Evaluator` object that you want to plot.
        return_fig: bool, optional, default=True
            Whether to return the figure or just print it.
        """
        fig = self.evaluator.plot(metric_set_name, *[log for log in self.scenario_logs.values()],
                                  labels=[k for k in self.scenario_logs.keys()], return_fig=True)
        if return_fig:
            return fig
        else:
            plt.show()

    def _determine_sample_size(self):
        # solve equation for number of observations (nobs)
        group_sizes = [len(self.factors[f].keys()) for f in self.factors.keys()]
        n = self._solve_power(np.max(group_sizes))
        progress("Statistically desired runs: {}".format(n))
        return self._get_final_n_runs(n, k_groups=group_sizes, find_lcm=True)

    def save_simulation_logs(self, dirpath=".", prefix="experiment1"):
        """Write simulation logs to disk.

        Parameters
        ----------
        dirpath: str
            Path to directory to save the result in.
        prefix: str
            String to prepend to '_results_A' and 'results_B' for the two logs.
        """
        for nr, log in self.scenario_logs.items():
            log.to_csv(os.path.join(dirpath, prefix + "_scenario_{}.csv".format(nr)))

        with open(os.path.join(dirpath, prefix + "_scenario_explanation.txt"), "w") as f:
            print("Scenario setups:", file=f)
            print("----------------\n", file=f)
            for k, dict_ in self.scenario_dict.items():
                print("Scenario {}".format(k), file=f)
                for key, value in dict_.items():
                    print("- " + str(key) + ": " + str(value), file=f)
                print("----", file=f)