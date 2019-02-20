import numpy as np
import pandas as pd
from scipy import stats
from fdsim.helpers import progress


class Evaluator(object):
    """Class that evaluates simulation runs, i.e., extracts metrics from the simulation log.

    Multiple metrics can be set up in once Evaluator object, so that all metrics are
    calculated upon every call to '.evaluate()'. This way, the evaluator only has to be
    initialized once in order to run simulation experiments with multiple input configurations.

    Parameters
    ----------
    response_time_col, target_col, run_col, prio_col, location_col, vehicle_col,
    incident_type_col, object_col: str, optional,
        The columns in the simulation log(s) that will refer to varying aspects of an incident
        or deployment. Specifically, the columns represent, respectively: the response time,
        the response time targets, the simulation run/iteration respectively, the priority
        of the incident, the demand location id of the incident, the vehicle type, the
        incident type, and the object function. Defaults are "response_time", "target", "run",
        "priority", "location", "vehicle_type", "incident_type", and "object_function"
        respectively.
    by_run: boolean, optional (default: True),
        Whether to calculate metrics per simulation run (True) or over the whole dataset.

    Notes
    -----
    The Evaluator class was developed with flexibility as one of the most important criteria.
    To support this flexibility, while maintaining a simple API, metrics are not defined upon
    initialization, but using the '.add_metric()' method.
    """

    measures = ["response_time", "on_time", "delay"]
    filters = ["locations", "prios", "vehicles", "incident_types", "objects"]

    def __init__(self, response_time_col="response_time", target_col="target", run_col="run",
                 prio_col="priority", location_col="location", vehicle_col="vehicle_type",
                 incident_type_col="incident_type", object_col="object_function",
                 incident_id_col="t", by_run=True, confidence=0.95, verbose=True):
        # storage of metrics to compute
        self.metric_sets = {}
        self.metric_set_names = []
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

        # column map for filtering
        self.filter_column_map = {"locations": self.loc_col,
                                  "prios": self.prio_col,
                                  "vehicles": self.vtype_col,
                                  "incident_types": self.itype_col,
                                  "objects": self.object_col}

        # other parameters
        self.by_run = by_run
        self.confidence = confidence
        self.verbose = verbose

    def add_metric(self, measure, name=None, description=None, mean=True, std=True,
                   missing=True, quantiles=[0.5, 0.75, 0.90, 0.95, 0.98, 0.99],  prios=None,
                   locations=None, vehicles=None, incident_types=None, objects=None,
                   first_only=False):
        """Add metrics that should be evaluated.

        Parameters
        ----------
        measure: str, one of ["response_time", "on_time", "delay"],
            The measure to evaluate.
        name: str, optional (default: None),
            How to name the set of metrics for reference in outputs. If None, a standard name
            is given (i.e., 'metric set 1', 'metric set 2').
        description: str, optional (default: None),
            A description of the set of evaluation metrics. This can be used to explain, e.g.,
            the applied filtering in a more elaborate way, whereas the 'name' property should
            be kept concise.
        mean, std, missing: boolean, optional (default: True),
            Whether to describe the measure by its mean, standard deviation and proportion of
            missing (NaN) values. Note that a missing response time means the response was
            carried out by an external vehicle.
        quantiles: array(float), optional (default: [0.5, 0.75, 0.90, 0.95, 0.98, 0.99]),
            Which quantiles to describe the measure with. Set to None to not use any quantiles.
        prios: int or array-like of ints, optional (default: None),
            Which priority levels to include during evaluation. If None, uses all levels.
        locations, vehicles, incident_types, objects: array(str), optional (default: None),
            Which locations, vehicles types, incident types and object functions to include
            during evaluation. If None, uses all values.
        first_only: boolean, optional (default: False),
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

        self.metric_sets[name] = {"mean": mean, "std": std, "missing": missing,
                                  "quantiles": quantiles, "locations": locations,
                                  "prios": prios, "vehicles": vehicles,
                                  "incident_types": incident_types, "objects": objects,
                                  "first_only": first_only, "description": description,
                                  "measure": measure}

        self.metric_set_names.append(name)
        progress("Set of metrics '{}' added.".format(name), verbose=self.verbose)

    def evaluate(self, log):
        """Evaluate a given simulation output on all set metrics.

        Parameters
        ----------
        log: pd.DataFrame,
            The raw simulation output/log.

        Returns
        -------
        metrics: pd.DataFrame,
            The calculated metrics.
        """
        progress("Evaluating {} sets of metrics.".format(len(self.metric_set_names)),
                 verbose=self.verbose)
        result_dict = {}
        for name in self.metric_set_names:
            progress("Evaluating {}.".format(name))
            result_dict[name] = self._evaluate_metric_set(log, self.metric_sets[name])

        progress("Evaluation completed.", verbose=self.verbose)
        return result_dict

    def _evaluate_metric_set(self, log, metric_set):
        """Evaluate a set of metrics relating to a single measure.

        Parameters
        ----------
        log: pd.DataFrame,
            The log of simulation outputs.
        metric_set: dict,
            The description of the metrics to calculate as created in '.add_metric()'.

        Returns
        -------
        result: pd.DataFrame,
            The calculated metrics.
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

        # calculate metrics
        progress("Calculating requested metrics.", verbose=self.verbose)
        results_per_run = self._calculate_descriptors_by_run(
            data,
            y_col=y_col,
            mean=metric_set["mean"],
            std=metric_set["std"],
            missing=metric_set["missing"],
            quantiles=metric_set["quantiles"]
        )

        results_per_run.drop("run", axis=1, inplace=True)
        return results_per_run

    @staticmethod
    def _filter_data(data, col, values):
        """Filter data while dealing with input variations."""
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

    def _calculate_descriptors_by_run(self, data, y_col, mean=True, std=True, missing=True,
                                      quantiles=None, measure_col="response_time"):
        """Calculate requested metrics for each simulation run."""
        results = (data.groupby(self.run_col)[y_col]
                       .apply(lambda x: self._calculate_descriptors(x, mean=mean, std=std,
                                                                    missing=missing,
                                                                    quantiles=quantiles)))

        return results.reset_index("run").reset_index(drop=True)

    def _calculate_descriptors(self, x, mean=True, std=True, missing=True, quantiles=None):
        """Calculate requested metrics over an array x."""
        length = len(x)
        x, n_missing = self._count_and_drop_nan(x)
        descriptors = {}

        if missing:
            descriptors["n_missing"] = n_missing
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

    def plot_distributions(self, data, metric_name):
        pass

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
