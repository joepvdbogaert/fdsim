import pandas as pd
import numpy as np
from abc import abstractmethod, ABCMeta

import statsmodels.api as sm
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

            for name, f in self.test_results.items():
                print("------------------------------------------------------------")
                print("Measure: '{}'\n{}"
                      .format(name, self.evaluator.metric_sets[name]["description"]))
                print("------------------------------")
                print(pd.DataFrame(f).T, end="\n\n")

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
