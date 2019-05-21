"""The :code:`fdsim` package has built-in functionality to analyze simulation log files and extract
useful performance measures. This functionality is provided in one flexible class that can be
configured to output a wide variety of performance indicators.

For example, you can filter the deployments or incidents that should be taken into account
when calculating performance and you can determine which descriptors, such as quantile values,
of a certain performance measure (e.g., the response time) you want to calculate. In addition,
you can configure the :code:`Evaluator` class to use multiple performance metrics and reuse the same
object among different simulation setups to compare the results.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.ticker import PercentFormatter, AutoMinorLocator, MultipleLocator
from scipy import stats
from fdsim.helpers import progress


class Evaluator(object):
    """Class that evaluates simulation runs, i.e., extracts metrics from the simulation log.

    Multiple metrics can be set up in one :code:`Evaluator` object, so that all metrics are
    calculated upon every call to :code:`Evaluator.evaluate`. This way, the evaluator only
    has to be initialized once in order to run simulation experiments with multiple input
    configurations.

    Parameters
    ----------
    response_time_col, target_col, run_col, prio_col, location_col, vehicle_col, incident_type_col, object_col: str, optional
        The columns in the simulation log(s) that will refer to varying aspects of an incident
        or deployment. Specifically, the columns represent, respectively: the response time,
        the response time targets, the simulation run/iteration respectively, the priority
        of the incident, the demand location id of the incident, the vehicle type, the
        incident type, and the object function. Defaults are "response_time", "target", "run",
        "priority", "location", "vehicle_type", "incident_type", and "object_function"
        respectively.
    by_run: boolean, optional, default=True
        Whether to calculate metrics per simulation run (True) or over the whole dataset.

    Notes
    -----
    The :code:`Evaluator` class was developed with flexibility as one of the most important criteria.
    To support this flexibility, while maintaining a simple API, metrics are not defined upon
    initialization, but using the :code:`.add_metric()` method.
    """

    measures = ["response_time", "on_time", "delay"]
    filters = ["locations", "prios", "vehicles", "incident_types", "objects", "hours",
               "days_of_week"]
    hour_col = "hour"
    weekday_col = "weekday"

    def __init__(self, response_time_col="response_time", target_col="target", run_col="run",
                 prio_col="priority", location_col="location", vehicle_col="vehicle_type",
                 incident_type_col="incident_type", object_col="object_function",
                 incident_id_col="t", datetime_col="time", by_run=True, confidence=0.95, verbose=True):

        # storage of metrics to compute
        self.metric_sets = {}
        self.metric_set_names = []
        self.metric_set_measures = {}

        # column names
        self.response_time_col = response_time_col
        self.target_col = target_col
        self.run_col = run_col
        self.prio_col = prio_col
        self.loc_col = location_col
        self.vtype_col = vehicle_col
        self.itype_col = incident_type_col
        self.object_col = object_col
        self.incident_id_col = incident_id_col
        self.datetime_col = datetime_col

        # column map for filtering
        self.filter_column_map = {"locations": self.loc_col,
                                  "prios": self.prio_col,
                                  "vehicles": self.vtype_col,
                                  "incident_types": self.itype_col,
                                  "objects": self.object_col,
                                  "hours": self.hour_col,
                                  "days_of_week": self.weekday_col}

        # other parameters
        self.by_run = by_run
        self.confidence = confidence
        self.verbose = verbose

    def add_metric(self, measure, name=None, description=None, count=True, mean=True, std=True,
                   missing=True, quantiles=[0.5, 0.75, 0.90, 0.95, 0.98, 0.99],  prios=None,
                   locations=None, vehicles=None, incident_types=None, objects=None,
                   hours=None, days_of_week=None, first_only=False):
        """Add metrics that should be evaluated.

        Parameters
        ----------
        measure: str, one of ["response_time", "on_time", "delay"]
            The measure to evaluate.
        name: str, optional, default=None
            How to name the set of metrics for reference in outputs. If None, a standard name
            is given (i.e., 'metric set 1', 'metric set 2').
        description: str, optional, default=None
            A description of the set of evaluation metrics. This can be used to explain, e.g.,
            the applied filtering in a more elaborate way, whereas the 'name' property should
            be kept concise.
        count, mean, std, missing: boolean, optional, default=True
            Whether to describe the measure by its count, mean, standard deviation and
            proportion of missing (NaN) values. Note that a missing response time means the
            response was carried out by an external vehicle.
        quantiles: array(float), optional, default=[0.5, 0.75, 0.90, 0.95, 0.98, 0.99])
            Which quantiles to describe the measure with. Set to None to not use any quantiles.
        prios: int or array-like of ints, optional, default=None
            Which priority levels to include during evaluation. If None, uses all levels.
        locations, vehicles, incident_types, objects: array(str), optional (default: None),
            Which locations, vehicles types, incident types and object functions to include
            during evaluation. If None, uses all values.
        hours: array-like of ints or None, optional, default=None
            Which hours of dat to incorporate during evaluation. Values must be integers in
            [0, 23].
        days_of_week: array-like of ints or None, optional, default=None
            Which days of the week to incorporate during evaluation. Monday = 0, ..., Sunday = 6.
        first_only: boolean, optional, default=False
            Whether to calculate the metrics for only the first arriving vehicle per incident
            (True) or to evaluate all vehicles (False).
        """
        if name is None:
            if len(self.metric_set_names) == 0:
                i = 1
            else:
                i = int(np.max([int(n[-1]) for n in self.metric_set_names]) + 1)
            name = "metric_set_{}".format(i)

        assert measure in self.measures, "'measure' must be one of {}. Received {}" \
            .format(measure, self.measures)
        self.metric_set_measures[name] = measure

        if locations is not None:
            locations = np.array(locations, dtype=str)

        self.metric_sets[name] = {"count": count, "mean": mean, "std": std, "missing": missing,
                                  "quantiles": quantiles, "locations": locations,
                                  "prios": prios, "vehicles": vehicles,
                                  "incident_types": incident_types, "objects": objects,
                                  "hours": hours, "days_of_week": days_of_week,
                                  "first_only": first_only, "description": description,
                                  "measure": measure}

        self.metric_set_names.append(name)
        progress("Set of metrics '{}' added.".format(name), verbose=self.verbose)

    def evaluate(self, log):
        """Evaluate a given simulation output on all set metrics.

        Parameters
        ----------
        log: pd.DataFrame
            The raw simulation output/log.

        Returns
        -------
        metrics: pd.DataFrame
            The calculated metrics.
        """
        progress("Evaluating {} sets of metrics.".format(len(self.metric_set_names)),
                 verbose=self.verbose)
        result_dict = {}
        for name in self.metric_set_names:
            progress("Evaluating {}.".format(name), verbose=self.verbose)
            result_dict[name] = self._evaluate_metric_set(log, self.metric_sets[name])

        progress("Evaluation completed.", verbose=self.verbose)
        return result_dict

    def _apply_filters(self, log, metric_set):
        """Applies all the filtering specified in a metric set and returns the resulting
        observations in the simulation log. Also adds relevant performance measures.

        Parameters
        ----------
        log: pd.DataFrame
            The simulation log
        metric_set: dict
            The metric set as created by this class.

        Returns
        -------
        data: pd.Dataframe
            The filtered log.
        y_col: str
            The name of the column that describes the measure of the metric set.
        """
        data = log.copy()
        # apply filters
        for f in self.filters:
            if metric_set[f] is not None:
                progress("Filtering on {}.".format(f), verbose=self.verbose)
                data = self._filter_data(data, self.filter_column_map[f], metric_set[f])

        if metric_set["first_only"]:
            progress("Keeping only first vehicle per incident.", verbose=self.verbose)
            data.sort_values(self.response_time_col, inplace=True)
            data.drop_duplicates(subset=[self.run_col, self.incident_id_col], inplace=True)
            data.sort_values([self.run_col, self.incident_id_col], inplace=True)

        # add relevant performance measures to data
        if metric_set["measure"] == "response_time":
            y_col = self.response_time_col
        if metric_set["measure"] == "on_time":
            data["on_time"] = (data[self.response_time_col] <= data[self.target_col])
            y_col = "on_time"
        if metric_set["measure"] == "delay":
            data["delay"] = data[self.response_time_col] - data[self.target_col]
            y_col = "delay"

        return data, y_col

    def _evaluate_metric_set(self, log, metric_set):
        """Evaluate a set of metrics relating to a single measure.

        Parameters
        ----------
        log: pd.DataFrame
            The log of simulation outputs.
        metric_set: dict
            The description of the metrics to calculate as created in :code:`.add_metric()`.

        Returns
        -------
        result: pd.DataFrame
            The calculated metrics.
        """
        data, y_col = self._apply_filters(log, metric_set)

        # calculate metrics
        progress("Calculating requested metrics.", verbose=self.verbose)
        if self.by_run:
            results_per_run = self._calculate_descriptors_by_run(
                data,
                y_col=y_col,
                count=metric_set["count"],
                mean=metric_set["mean"],
                std=metric_set["std"],
                missing=metric_set["missing"],
                quantiles=metric_set["quantiles"]
            )

            results_per_run.drop(self.run_col, axis=1, inplace=True)
            return results_per_run
        else:
            results = self._calculate_descriptors(
                data[y_col],
                count=metric_set["count"],
                mean=metric_set["mean"],
                std=metric_set["std"],
                missing=metric_set["missing"],
                quantiles=metric_set["quantiles"]
            )
            return results

    def _filter_data(self, data, col, values):
        """Filter data while dealing with input variations."""
        if col == self.hour_col:
            data[self.hour_col] = pd.to_datetime(data[self.datetime_col]).dt.hour
        if col == self.weekday_col:
            data[self.weekday_col] = pd.to_datetime(data[self.datetime_col]).apply(
                lambda x: x.isoweekday())

        if isinstance(values, (list, np.ndarray, pd.Series)):
            if len(values) > 1:
                data = data[np.in1d(data[col], values)].copy()
            elif len(values) == 1:
                values = values[0]
            else:
                raise ValueError("'values' cannot be empty. Received: {}.".format(values))
        else:  # 'values' is a single value (str, float, int, etc.)
            data = data[data[col] == values].copy()
        return data

    def _calculate_descriptors_by_run(self, data, y_col, count=True, mean=True, std=True,
                                      missing=True, quantiles=None,
                                      measure_col="response_time"):
        """Calculate requested metrics for each simulation run."""
        results = (data.groupby(self.run_col)[y_col]
                       .apply(lambda x: self._calculate_descriptors(x, count=count, mean=mean,
                                                                    std=std, missing=missing,
                                                                    quantiles=quantiles)))

        return results.reset_index(self.run_col).reset_index(drop=True)

    def _calculate_descriptors(self, x, count=True, mean=True, std=True, missing=True,
                               quantiles=None):
        """Calculate requested metrics over an array x."""
        length = len(x)
        x, n_missing = self._count_and_drop_nan(x)
        descriptors = {}

        if count:
            descriptors["count"] = length

        if missing:
            # descriptors["n_missing"] = n_missing
            descriptors["prop_missing"] = n_missing / length

        if mean:
            descriptors["mean"] = np.mean(x)

        if std:
            descriptors["std"] = np.std(x)

        if quantiles is not None:
            for q in quantiles:
                descriptors["{}-quantile".format(q)] = np.quantile(x, q)

        return pd.DataFrame(descriptors, index=[x.name])

    def _get_confidence_intervals_per_column(self, data):

        N = len(data)
        means, stds = data.mean(axis=0), data.std(axis=0)
        df = pd.DataFrame({"mean": means, "std": stds}, index=means.index)
        df[["LB", "UB"]] = (
            df.apply(lambda x: stats.norm.interval(self.confidence, loc=x["mean"],
                                                   scale=x["std"]/np.sqrt(N)),
                     axis=1)
              .apply(pd.Series)
            )
        return df

    @staticmethod
    def _count_and_drop_nan(x):
        number_nan = np.sum(np.isnan(x))
        filtered_x = x[~np.isnan(x)]
        return filtered_x, number_nan

    def plot(self, metric_set_name, *datasets, return_fig=True, labels=None, **kwargs):
        """Plot the distributions of a measure in various simulation results logs.

        Parameters
        ----------
        metric_set_name: str
            The name of the metric set to plot. Filters are applied specified for this metric
            set. Note that metrics are not computed, but the filtered measure is plotted as
            a continous variable. Hence, the measure of the metric set should be either
            response time or delay and cannot be 'on time'.
        *datasets: pd.DataFrames
            Simulation logs from experiments that should be plotted in the same chart.
        return_fig: boolean, optional, default=True
            Whether to return the figure object (True) or to plot directly (False).
        """
        if labels is None:
            labels = ["Scenario {}".format(i) for i in range(len(datasets))]

        sns.set()
        plt.rcParams['xtick.bottom'] = True
        plt.rcParams['ytick.left'] = True
        plt.rc('font', size=18)          # controls default text sizes
        # plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
        plt.rc('axes', labelsize=16)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=14)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=14)    # fontsize of the tick labels
        plt.rc('legend', fontsize=16)    # legend fontsize
        plt.rc('figure', titlesize=20) 
        # plt.rcParams.update({'font.size': 18})

        # apply filtering to the logs according to metric set
        filtered_datasets = []
        for dataset in datasets:
            tmp, y_col = self._apply_filters(dataset, self.metric_sets[metric_set_name])
            filtered_datasets.append(tmp)

        # drop NaNs
        filtered_datasets = [data.dropna(subset=[y_col]) for data in filtered_datasets]

        # TODO: determine tail start and end for both plots
        if y_col == "response_time":
            tail_start_0 = 0
        else:
            tail_start_0 = np.min([np.quantile(data[y_col], q=0.002) for data in filtered_datasets])

        tail_end_0 = np.max([np.quantile(data[y_col], q=0.998) for data in filtered_datasets])
        tail_start_1 = np.min([np.quantile(data[y_col], q=0.95) for data in filtered_datasets])
        tail_end_1 = np.max([np.quantile(data[y_col], q=0.999) for data in filtered_datasets])
        
        # plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0] = self._exceed_plots(*filtered_datasets, y_col=y_col, run_col=self.run_col,
                                     tail_start=tail_start_0, tail_end=tail_end_0, ax=axes[0], labels=labels,
                                     **kwargs)
        axes[1] = self._exceed_plots(*filtered_datasets, y_col=y_col, run_col=self.run_col,
                                     tail_start=tail_start_1, tail_end=tail_end_1, ax=axes[1], labels=labels,
                                     **kwargs)
        fig.suptitle("Probabilities of exceeding values for '{}'".format(metric_set_name))
        fig.tight_layout()
        fig.subplots_adjust(top=0.9)
        if return_fig:
            return fig
        else:
            plt.show()

    def _calc_cumulative_probs(self, data, y_col="response_time", run_col="run", tail_start=800,
                               tail_end=2500, new_x="response time",
                               new_y="probability to exceed"):
        datasets = []
        for run in data[run_col].unique():
            df = pd.DataFrame({new_x: np.arange(tail_start, tail_end, 10)})
            df[run_col] = run
            arr = data[data[run_col] == run][y_col].values
            df[new_y] = df[new_x].apply(lambda x: np.mean(arr > x))
            datasets.append(df)
        result = pd.concat(*[datasets], axis=0)
        return result

    def _exceed_plot(self, data, y_col="response_time", run_col="run", tail_start=800,
                     tail_end=2500, ax=None, new_x="response time",
                     new_y="probability to exceed", label=None):
        df_long = self._calc_cumulative_probs(data, y_col=y_col, run_col=run_col, tail_start=tail_start,
                                              tail_end=tail_end)
        ax = sns.lineplot(x=new_x, y=new_y, data=df_long, ax=ax, estimator="mean", ci="sd", label=label, legend="full")
        ax.set_xlim(tail_start, tail_end)
        return ax

    def _exceed_plots(self, *datasets, y_col="response_time", run_col="run", ax=None,
                     tail_start=800, tail_end=2500, new_x="response time",
                      new_y="probability to exceed", labels=None):
        if labels is None:
            labels = ["Scenario {}" for i in range(len(datasets))]

        if ax is None:
            fig, ax = plt.subplots(figsize=(6,6))
        for i, data in enumerate(datasets):
            ax = self._exceed_plot(data, y_col=y_col, ax=ax, tail_start=tail_start, tail_end=tail_end, new_x=new_x, new_y=new_y, label=labels[i])

        if y_col == "response_time":
            ax.set_xlabel("response time (seconds)")
        else:
            ax.set_xlabel("delay (seconds)")

        ax.tick_params(direction="out", colors="black", length=5, width=1)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
        minorLocator = MultipleLocator(5)
        ax.yaxis.set_minor_locator(minorLocator)
        ax.minorticks_on()
        return ax

    @staticmethod
    def _to_long_format(data, drop_cols=["run", "n_missing"], colnames=["metric", "value"]):
        """Transform metric data to long format for plotting."""
        df_long = (data.drop(drop_cols, axis=1)
                       .T
                       .stack()
                       .reset_index(level=1, drop=True)
                       .reset_index())

        df_long.columns = colnames
        return df_long
