import pandas as pd
import numpy as np
from abc import abstractmethod, ABCMeta

import statsmodels.api as sm
import statsmodels.formula.api as smf

from fractions import gcd
from functools import reduce

from fdsim.helpers import progress
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
        """Add effect sizes eta^2 and omega^2 to an ANOVA table from statsmodels."""
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

    def print_results(self):
        if self.test_results is None:
            progress("Nothing to print.")
        else:
            print("------------------\nA/B Test Results{}\n------------------"
                  .format(": " + self.name or ""))
            if self.description is not None:
                print(self.description)

            if self.description_A is not None and self.description_B is not None:
                print("Scenario A: {}".format(self.description_A))
                print("Scenario B: {}".format(self.description_B))
                print("------------------", end="\n\n")
            for name, f in self.test_results.items():
                print("Measure: '{}'\n{}"
                      .format(name, self.evaluator.metric_sets[name]["description"]))
                print("------------------------------------------------------------")
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

        # for debugging:
        self.results_dict = results_dict

        progress("Simulation completed. Conducting statistical analysis on results.")
        self.test_results = self.analyze(results_dict)
        progress("Statistical tests performed and results obtained.")

    def analyze(self, results_dict):
        """Analyze the results using one-way ANOVA."""
        # reorganize the data by measure and add scenario as column
        metric_data = {}
        for scenario, scenario_results in results_dict.items():
            for metric_set, results in scenario_results.items():
                results["scenario"] = scenario
                if metric_set in metric_data.keys():  # concat on previous data
                    metric_data[metric_set] = metric_data[metric_set].append(results, ignore_index=True)
                else:  # create new entry in dictionary
                    metric_data[metric_set] = results

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

    def print_results(self):
        """Print the results of the ANOVA and Tukey analysis to give a quick overview of the
        results."""
        if (self.anova_results is None) and (self.tukey_results is None):
            progress("Nothing to print.")
        else:
            print("------------------\nMultiple Scenarios Test Results{}\n------------------"
                  .format(": " + self.name or ""))

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
                print(aov_table)
                print("------------------------------------------------------------", end="\n\n")

                print("Tukey HSD post-hoc analysis: pairwise comparison of relevant groups")
                print("------------------------------------------------------------")
                print(self.tukey_results[measure])
                print("------------------------------------------------------------", end="\n\n")
