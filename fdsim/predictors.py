import os
import warnings
from abc import abstractmethod, ABCMeta

import numpy as np
import pandas as pd

from fbprophet import Prophet
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error


class BaseIncidentPredictor(object):
    """ Base class for incident predictors. Not useful to instantiate
        on its own.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def fit(self, data):
        """ Fit the model on the data. """

    @abstractmethod
    def predict(self, data):
        """ Predict using the fitted model. """

    @staticmethod
    def evaluate(y_true, y_predict, metric="RMSE"):
        """ Evaluate a given prediction.

        Parameters
        ----------
        y_true: array,
            The ground truth values.
        y_predict: array
            The predicted labels / forecasted values.
        metric: str, one of ['MAE', 'RMSE'], optional (default: 'RMSE'),
            The evaluation metric. Uses the Root Mean Squared Error for 'RMSE' and the
            Mean Absolute Error for 'MAE'.

        Returns
        -------
        score: float,
            The error score(s) of the prediction. If y is multi-output, outputs a list of
            scores (one score per variable).
        """
        assert len(y_true) == len(y_predict), \
            "Values and predictions must have same length"
        if metric == "RMSE":
            return np.sqrt(mean_squared_error(y_true, y_predict, multioutput="raw_values"))
        elif metric == "MAE":
            return mean_absolute_error(y_true, y_predict, multioutput="raw_values")
        else:
            raise ValueError("{} is not a supported value for 'metric'. "
                             "Must be one of ['RMSE', 'MAE'].")

    @abstractmethod
    def _create_sampling_dict(self):
        """ Create a dictionary to sample incidents from. """

    @staticmethod
    def _infer_time_unit(time_sequence):
        x = pd.to_datetime(pd.Series(time_sequence))
        unit = (x.max() - x.min()) / (len(x)-1)
        return unit

    def save_forecast(self):
        """ Save forecasted incident rate to csv. """
        self.forecast.to_csv(os.path.join(self.fc_dir, self.file_name),
                             index=False)

    def get_forecast(self):
        """ Return the DataFrame with the forecast. """
        return self.forecast

    def get_sampling_dict(self):
        """ Return the dictionary from which to sample. """
        if self.sampling_dict is None:
            self._create_sampling_dict()
        return self.sampling_dict

    @staticmethod
    def _create_date_hour_column(data, datetime_col="dim_incident_start_datumtijd"):
        """ Create a datetime column with hourly precision to facilitate counting incidents
        per hour. """
        dts = pd.to_datetime(data[datetime_col])
        return dts.apply(lambda x: pd.Timestamp(year=x.year, month=x.month,
                                                day=x.day, hour=x.hour))

    @staticmethod
    def _create_complete_hourly_index(start_datetime, end_datetime=None, n_hours=None):
        """ Create an array of datetime values of hourly precision from start to end or from
        start until a specified number of values. The array can be used to reindex a dataframe
        that counts incidents per hour, so that values of zero will be listed as well.
        """
        start_datetime = pd.to_datetime(start_datetime)
        end_datetime = pd.to_datetime(end_datetime)
        if end_datetime is not None:
            result = pd.date_range(start=start_datetime, end=end_datetime, freq="H")
        elif n_hours is not None:
            result = pd.date_range(start=start_datetime, end=end_datetime, periods=n_hours, freq="H")
        else:
            raise ValueError("One of 'end_datetime' or 'n_hours' must be given")

        index = pd.Index(result, name="hourly_index")
        return index

    def ts_cross_validate(self, data, n_splits=5, types=None, last_n_years=True, metric="MAE"):
        """ Perform n-fold time series cross validation to evaluate the forecast method. """
        if types is not None:
            data = data[np.in1d(data["dim_incident_incident_type"], types)].copy()
        else:
            data = data[~np.in1d(data["dim_incident_incident_type"], ["nan", "NVT", np.nan])].copy()

        print(data.shape)
        data["datetime"] = self._create_date_hour_column(data)
        print("min, max time: {}, {}".format(data["datetime"].min(), data["datetime"].max()))
        times = self._create_complete_hourly_index(data["datetime"].min(), end_datetime=data["datetime"].max())
        if types is None:
            types = list(set(data["dim_incident_incident_type"].unique()) - set(["nan", "NVT", np.nan]))

        ts = pd.pivot_table(
            data,
            values="dim_incident_id",
            index="datetime",
            columns="dim_incident_incident_type",
            aggfunc="count",
            fill_value=0
        )
        ts = ts.reindex(times, fill_value=0, axis=0)
        assert len(ts) == len(times), \
            "Pivot went wrong, length pivot = {}, while length times = {}".format(len(ts), len(times))

        if last_n_years:
            splitter = YearSplitter(n_splits=n_splits, obs_per_year=365*24)
        else: # use classic equal-sized folds
            splitter = TimeSeriesSplit(n_splits=n_splits)

        scores = []
        for train_index, test_index in splitter.split(times):
            print("train index: {} ({}) - {} ({})".format(times[train_index[0]], np.min(train_index), times[train_index[-1]], np.max(train_index)))
            print("test index: {} ({}) - {} ({})".format(times[test_index[0]], np.min(test_index), times[test_index[-1]], np.max(test_index)))
            future = pd.DataFrame({"ds": times[test_index]})

            self.fit(
                data[(data["datetime"] >= times[train_index[0]]) & (data["datetime"] < times[train_index[-1]])],
                types=types
            )
            self.predict(future=future)
            y_predict = self.get_forecast().drop("ds", axis=1)
            y_true = ts.iloc[test_index, :]
            scores.append(self.evaluate(y_true, y_predict, metric=metric))

        return np.array(scores)


class ProphetIncidentPredictor(BaseIncidentPredictor):
    """ Class that forecasts incident rate for different incident types.

    Uses Facebook's Prophet to create a forecast of the incident rate.
    It does so by calculating the hourly arrivals per incident type, then
    treating this as a signal/time series and decomposing it into trend,
    yearly pattern, weekly pattern, and daily pattern.

    Example
    -------
    >>> predictor = ProphetIncidentPredictor(load_forecast=False)
    >>> predictor.fit(incident_data)
    >>> predictor.predict(periods=365*24, freq="H", save=True)
    >>> forecast = predictor.get_forecast()
    >>> forecast.head()

    Parameters
    ----------
    load_forecast: boolean
        Whether to load a pre-existing forecast from disk.
        Defaults to True, since recomputing forecasts is costly.
    fc_dir: str
        The directory in which forecasts should be saved and
        from which they should be loaded if applicable. Defaults
        to './data/forecasts/'.
    verbose: boolean
        Whether to print what is happening, defaults to True.
    """

    __name__ = "ProphetIncidentPredictor"

    def __init__(self, load_forecast=True, fc_dir="data/forecasts",
                 verbose=True):

        self.verbose = verbose
        self.fitted = False
        self.file_name = "prophet_forecast.csv"
        self.fc_dir = fc_dir
        self.forecast = None
        self.sampling_dict = None

        if load_forecast:
            try:
                self.forecast = pd.read_csv(os.path.join(self.fc_dir,
                                                         self.file_name))
                self.types = [c for c in self.forecast.columns if c != "ds"]
            except FileNotFoundError:
                warnings.warn("No forecast found, check if 'fc_dir' specifies"
                              " the right directory. If you didn't mean to "
                              "load a forecast, initialize with "
                              "'load_forecast=False'. Instance intialized "
                              "without forecast. Create one by running "
                              "IncidentPredictor.fit() and then .predict()."
                              "Given directory: {}.".format(self.fc_dir))

    def fit(self, data, types=None):
        """ Perform time series decomposition using Prophet.

        This function first prepares the data and saves the prepared data
        as 'self.incidents'. then it creates a dictionary of Prophet() objects,
        where the keys equal the incident types and the corresponding model
        is fitted to the data of that type. The dictionary of models is stored
        as 'self.models_dict' and used when predict is called.

        Notes
        -----
        This function does not return anything.

        Parameters
        ----------
        data: pd.DataFrame
            The incidents to train the models on.
        types: Sequence(str)
            The incident types to fit models for. If None, uses
            all incident types in the data, except 'nan' and 'NVT'.
            Defaults to None.
        """

        if types is not None:
            self.types = types
        else:
            if self.verbose:
                print("No incident types specified, using all types in data.")

            self.types = [t for t in data["dim_incident_incident_type"]
                          .unique() if t not in ["nan", "NVT", np.nan]]

        if self.verbose:
            print("Preparing incident data for analysis...")

        self.incidents = self._prep_data_for_prediction(data)
        self.incidents["hourly_datetime"] = self._create_date_hour_column(
            self.incidents,
            datetime_col="dim_incident_start_datumtijd"
        )

        start = self.incidents["hourly_datetime"].min()
        end = self.incidents["hourly_datetime"].max()
        self.time_index = self._create_complete_hourly_index(start, end_datetime=end)

        self.models_dict = dict()
        for type_ in self.types:
            if self.verbose:
                print("Fitting model for type {}...".format(type_))

            m = Prophet()
            dfprophet = self._create_prophet_data(self.incidents, self.time_index, type_=type_)
            m.fit(dfprophet)
            self.models_dict[type_] = m

        self.fitted = True
        if self.verbose:
            print("Models fitted.")

    def predict(self, periods=365*24, freq="H", save=False, future=None):
        """ Forecast the incident rate using Prophet.

        Notes
        -----
        Can only be called after calling '.fit()', throws assertion error
        otherwise. Does not return anything, since it's main use cases are
        sampling from directly from this predictor and saving predictions to
        file. The result of this method can be obtained by calling
        'get_forecast()' afterwards.

        Parameters
        ----------
        periods: int
            The number of periods to forecast.
        freq: str,
            The frequency to predict the incident rates at. Accepts any valid frequency
            for pd.date_range, such as 'H' (default), 'D', or 'M'.
        save: boolean
            Whether to save the forecast to a csv file. Optional, defaults to false.
        """
        assert self.fitted, "First use 'fit()' to fit a model before predicting."
        if future is None:
            future = self.models_dict[self.types[0]].make_future_dataframe(
                periods=periods, freq=freq, include_history=False)

        forecast_dict = dict(ds=future["ds"].tolist())

        for type_ in self.types:
            if self.verbose:
                print("Predicting incident rates for {}".format(type_))

            forecast_dict[type_] = np.maximum(0.0, self.models_dict[type_]
                                              .predict(future)["yhat"]
                                              .tolist())

        self.forecast = pd.DataFrame(forecast_dict)

        msg = ""
        if save:
            self.save_forecast()
            msg = " and saved to " + os.path.join(self.fc_dir, self.file_name)
        if self.verbose:
            print("Predictions made" + msg + ".")

    def _prep_data_for_prediction(self, incidents):
        """ Format time columns.

        Parameters
        ----------
        incidents: pd.DataFrame
            The incident data to prepare.

        Returns
        -------
        The prepared DataFrame and a pd.Index with hourly timestamps.
        """
        incidents["dim_tijd_uur"] = (incidents["dim_tijd_uur"].astype(float)
                                                              .astype(int)
                                                              .astype(str)
                                                              .str.zfill(2)
                                                              .copy())
        incidents["dim_datum_datum"] = pd.to_datetime(
            incidents["dim_datum_datum"]).dt.strftime("%Y-%m-%d").copy()
        # assert that it's sorted
        incidents.sort_values(["dim_datum_datum", "dim_tijd_uur"],
                              ascending=True, inplace=True)
        return incidents

    def _create_prophet_data(self, incidents, new_index, type_=None, groupby_col="hourly_datetime"):
        """ Create a DataFrame in the format required by Prophet.

        Parameters
        ----------
        incidents: pd.DataFrame
            The incident data.
        new_index: pandas.Index object
            Specifies the times that the resulting DataFrame should contain.
        type_: str or None (default: None)
            The incident type to make a DataFrame for. If None, ignore incident types.

        Returns
        -------
        A DataFrame with two columns: 'ds' and 'y', where 'ds' is the timestamp
        and 'y' is the number of incidents per time_unit. This DataFrame can be
        used directly as input for Prophet.fit().
        """
        dfprophet = incidents[["dim_incident_id", groupby_col, "dim_incident_incident_type"]].copy()
        if type_ is not None:
            dfprophet = dfprophet[dfprophet["dim_incident_incident_type"] == type_]
        dfprophet = dfprophet.groupby(groupby_col)["dim_incident_id"].size()
        dfprophet = dfprophet.reindex(new_index, fill_value=0).reset_index()
        dfprophet.rename(columns={"dim_incident_id": "y"}, inplace=True)
        dfprophet["ds"] = dfprophet[new_index.name]

        return dfprophet[["ds", "y"]]

    def create_sampling_dict(self, start_time=None, end_time=None, incident_types=None):
        """ Create a dictionary that can conveniently be used for
            sampling random incidents based on the forecast.

        Notes
        -----
        Stores three results:
            - self.sampling_dict, a dictionary like:

              {t -> {'type_distribution' -> probs,
                     'beta' -> expected interarrival time in minutes,
                     'time' -> the timestamp corresponding to start_time+t}
              }

              where t is an integer representing the time_units since the
              start_time.
            - self.sampling_start_time, timestamp of earliest time
              in the dictionary.
            - self.sampling_end_time, timestamp of the latest time
              in the dictionary.

        Parameters
        ----------
        start_time: Timestamp or str convertible to Timestamp
            The earliest time that should be included in the dictionary.
        end_time: Timestamp or str convertible to Timestamp
            The latest time that should be included in the dictionary.
        incident_types: array-like of strings
            The incident types to forecast for. Defaults to None. If None,
            uses all incident types in the forecast.

        Returns
        -------
        The sampling dictionary as described above.
        """
        assert self.forecast is not None, \
            ("No forecast available, initiate with load_forecast=True "
             "or use .fit() and .predict() to create one.")

        if incident_types is not None:
            fc = self.forecast[["ds"] + list(incident_types)].copy()
        else:
            fc = self.forecast.copy()

        fc["ds"] = pd.to_datetime(fc["ds"], dayfirst=True)
        if start_time is None:
            start_time = fc["ds"].min()
        if end_time is None:
            end_time = fc["ds"].max()

        fc = fc[(fc["ds"] >= start_time) & (fc["ds"] <= end_time)]
        time_unit = self._infer_time_unit(fc["ds"])
        del fc["ds"]

        rates_dict = fc.reset_index(drop=True).T.to_dict(orient="list")
        for i in rates_dict.keys():
            rts = rates_dict[i]
            rates_dict[i] = {"type_distribution": np.array(rts) / np.sum(rts),
                             "beta":  1 / np.sum(rts) * 60,
                             "time": start_time + i*time_unit}

        self.sampling_dict = rates_dict
        self.sampling_start_time = start_time or fc["ds"].min()
        self.sampling_end_time = end_time or fc["ds"].max()

        return rates_dict


class YearSplitter():
    """ Split data on whole years to provide constant evaluation metric. """

    def __init__(self, n_splits=3, obs_per_year=365*24):
        self.n = n_splits
        self.obs_per_year = obs_per_year

    def split(self, data):
        N = len(data)
        splits = [tuple([np.arange(0, N - self.obs_per_year * i),
                         np.arange(N - self.obs_per_year * i, N - self.obs_per_year*(i - 1))])
                  for i in range(1, self.n + 1)]

        for train, test in splits[::-1]:
            yield train, test
