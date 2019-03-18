import os
import warnings
from abc import abstractmethod, ABCMeta

import numpy as np
import pandas as pd

from fbprophet import Prophet
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error

from fdsim.helpers import progress


class BaseIncidentPredictor(object):
    """Base class for incident predictors. Not useful to instantiate on its own."""
    __metaclass__ = ABCMeta

    def __init__(self, load_forecast=True, fc_dir="data/forecasts", verbose=True):
        self.verbose = verbose
        self.fc_dir = fc_dir
        self.forecast = None

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
                              "self.fit() and then .predict()."
                              "Given directory: {}.".format(self.fc_dir))

    @abstractmethod
    def fit(self, data):
        """ Fit the model on the data. """

    @abstractmethod
    def predict(self, data):
        """ Predict using the fitted model. """

    @staticmethod
    def evaluate(y_true, y_predict, metric="RMSE"):
        """Evaluate a given prediction.

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

    def create_sampling_dict(self, start_time=None, end_time=None, incident_types=None):
        """Create a dictionary that can conveniently be used for
        sampling random incidents based on the forecast.

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
        sampling_dict: dict,
            The sampling dictionary as described below.

        Notes
        -----
        Stores three results:
            -self.sampling_dict, a dictionary like:
             `{t -> {'type_distribution' -> probs,
             'beta' -> expected interarrival time in minutes,
             'time' -> the timestamp corresponding to start_time+t}}`
             where t is an integer representing the time_units since the
             start_time.
            -self.sampling_start_time, timestamp of earliest time
             in the dictionary.
            -self.sampling_end_time, timestamp of the latest time
             in the dictionary.

        """
        assert self.forecast is not None, \
            ("No forecast available, initiate with load_forecast=True "
             "or use .fit() and .predict() to create one.")

        # determine incident types
        if incident_types is not None:
            fc = self.forecast[["ds"] + list(incident_types)].copy()
        else:
            fc = self.forecast.copy()

        # determine start and end times
        fc["ds"] = pd.to_datetime(fc["ds"], dayfirst=True)
        if start_time is None:
            start_time = fc["ds"].min()
        if end_time is None:
            end_time = fc["ds"].max()

        msg = "Creating a sampling dictionary from {} to {}.".format(start_time, end_time)
        progress(msg, verbose=self.verbose)

        # process date time range and remove it from the forecast
        fc = fc[(fc["ds"] >= start_time) & (fc["ds"] <= end_time)]
        timestamps = fc["ds"].copy()
        del fc["ds"]

        # create the dictionary
        rates_dict = fc.reset_index(drop=True).T.to_dict(orient="list")
        self.sampling_dict = {}
        for i, rts in rates_dict.items():
            self.sampling_dict[i] = {"type_distribution": np.array(rts) / np.sum(rts),
                                     "beta":  1 / np.sum(rts) * 60,
                                     "lambda": np.sum(rts),
                                     "time": timestamps.iloc[i]}

        # save start and end time for future reference
        self.sampling_start_time = start_time
        self.sampling_end_time = end_time

        progress("Sampling dictionary created.", verbose=self.verbose)
        return self.sampling_dict

    @staticmethod
    def _infer_time_unit(time_sequence):
        x = pd.to_datetime(pd.Series(time_sequence))
        unit = (x.max() - x.min()) / (len(x)-1)
        return unit

    def save_forecast(self):
        """ Save forecasted incident rate to csv. """
        path = os.path.join(self.fc_dir, self.file_name)
        self.forecast.to_csv(path, index=False)
        progress("Forecast saved to {}.".format(path), verbose=self.verbose)

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
            result = pd.date_range(start=start_datetime, end=end_datetime,
                                   periods=n_hours, freq="H")
        else:
            raise ValueError("One of 'end_datetime' or 'n_hours' must be given")

        index = pd.Index(result, name="hourly_index")
        return index

    def ts_cross_validate(self, data, n_splits=5, types=None, last_n_years=True, metric="MAE"):
        """ Perform n-fold time series cross validation to evaluate the forecast method. """
        if types is not None:
            data = data[np.in1d(data["dim_incident_incident_type"], types)].copy()
        else:
            data = (data[~np.in1d(data["dim_incident_incident_type"], ["nan", "NVT", np.nan])]
                    .copy())

        print(data.shape)
        data["datetime"] = self._create_date_hour_column(data)
        print("min, max time: {}, {}".format(data["datetime"].min(), data["datetime"].max()))
        times = self._create_complete_hourly_index(data["datetime"].min(),
                                                   end_datetime=data["datetime"].max())
        if types is None:
            types = list(set(data["dim_incident_incident_type"].unique()) -
                         set(["nan", "NVT", np.nan]))

        ts = pd.pivot_table(
            data,
            values="dim_incident_id",
            index="datetime",
            columns="dim_incident_incident_type",
            aggfunc="count",
            fill_value=0
        )
        ts = ts.reindex(times, fill_value=0, axis=0)
        assert len(ts) == len(times), "Pivot went wrong, length pivot = {}, " \
            "while length times = {}".format(len(ts), len(times))

        if last_n_years:
            splitter = YearSplitter(n_splits=n_splits, obs_per_year=365*24)
        else:  # use classic equal-sized folds
            splitter = TimeSeriesSplit(n_splits=n_splits)

        scores = []
        for train_index, test_index in splitter.split(times):

            future = pd.DataFrame({"ds": times[test_index]})

            self.fit(
                data[(data["datetime"] >= times[train_index[0]]) &
                     (data["datetime"] < times[train_index[-1]])],
                types=types
            )
            self.predict(future=future)
            y_predict = self.get_forecast().drop("ds", axis=1)
            y_true = ts.iloc[test_index, :]
            scores.append(self.evaluate(y_true, y_predict, metric=metric))

        return np.array(scores)

    def set_custom_forecast(self, forecast):
        self.forecast = forecast


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

    def __init__(self, **kwargs):

        self.fitted = False
        self.file_name = "prophet_forecast.csv"
        self.sampling_dict = None
        super().__init__(**kwargs)

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
            progress("No incident types given, using all types in data.", verbose=self.verbose)

            self.types = [t for t in data["dim_incident_incident_type"]
                          .unique() if t not in ["nan", "NVT", np.nan]]

        progress("Preparing incident data for analysis...", verbose=self.verbose)

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
            progress("Fitting model for type {}...".format(type_), verbose=self.verbose)

            m = Prophet()
            dfprophet = self._create_prophet_data(self.incidents, self.time_index, type_=type_)
            m.fit(dfprophet)
            self.models_dict[type_] = m

        self.fitted = True
        progress("Models fitted.", verbose=self.verbose)

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
            progress("Predicting incident rates for {}".format(type_), verbose=self.verbose)

            forecast_dict[type_] = np.maximum(0.0, self.models_dict[type_]
                                              .predict(future)["yhat"]
                                              .tolist())

        self.forecast = pd.DataFrame(forecast_dict)

        progress("Forecast made.", verbose=self.verbose)
        if save:
            self.save_forecast()

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

    def _create_prophet_data(self, incidents, new_index, type_=None,
                             groupby_col="hourly_datetime"):
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
        cols = ["dim_incident_id", groupby_col, "dim_incident_incident_type"]
        dfprophet = incidents[cols].copy()
        if type_ is not None:
            dfprophet = dfprophet[dfprophet["dim_incident_incident_type"] == type_]
        dfprophet = dfprophet.groupby(groupby_col)["dim_incident_id"].size()
        dfprophet = dfprophet.reindex(new_index, fill_value=0).reset_index()
        dfprophet.rename(columns={"dim_incident_id": "y"}, inplace=True)
        dfprophet["ds"] = dfprophet[new_index.name]

        return dfprophet[["ds", "y"]]


class BasicLambdaForecaster(BaseIncidentPredictor):
    """Forecast arrival rates of incidents based on historic averages.

    Arrival rates are obtained for every hour in the week, per month, per type of incident.
    So, different weeks in the same month always get the same arrival rates, but weeks in
    different months have different rates. Rates are determined as the average number of
    arrivals in a similar period.

    For example, the rate for a Monday in January between 8:00 and 9:00 is calculated as the
    average number of incidents between 8:00 and 9:00 of all Mondays in January in the time
    range of the data.

    Parameters
    ----------
    ignore_dates: array-like of datetime objects,
        Dates that are considered 'out of the ordinary' in terms of number of incidents
        and should not be taken into account when calculating average incident rates.
        Typically, this list includes days with storms and impactful events such as
        New Year's Eve and perhaps Kingsday.
    id_col, date_col, month_col, day_name_col, hour_col: str, optional,
        The column names indicating respectively the id of the incident, the date,
        month number, name of the week day, hour of day in [0, 24).
    **kwargs: dict,
        Parameters passed to BaseIncidentPredictor.
    """

    def __init__(self, ignore_dates=None, id_col="dim_incident_id",
                 type_col="dim_incident_incident_type", date_col="dim_datum_datum",
                 month_col="dim_datum_maand_nr", month_day_col="dim_datum_maand_dag_nr",
                 day_name_col="dim_datum_dag_naam_nl", hour_col="dim_tijd_uur", **kwargs):

        # store names of columns for use in multiple methods.
        self.id_col = id_col
        self.type_col = type_col
        self.date_col = date_col
        self.month_col = month_col
        self.day_name_col = day_name_col
        self.hour_col = hour_col
        self.month_day_col = month_day_col

        # save dates to ignore during modeling
        if ignore_dates is not None:
            ignore_dates = np.array(ignore_dates)
            if isinstance(ignore_dates[0], str):
                self.ignore_dates = pd.to_datetime(ignore_dates).values
                print("Found string values in 'ignore_dates'. I've converted them to datetime,"
                      " but it's safer to provide datetime objects in the first place.")
            else:
                self.ignore_dates = ignore_dates
        else:
            self.ignore_dates = None

        # fixed attributes
        self.lambdas = None
        self.fitted = False
        self.day_col = "weekday_number"
        self.file_name = "basic_lambda_forecast.csv"

        super().__init__(**kwargs)

    def fit(self, data, last_n_years=8, fit_nye=True):
        """Obtain arrival rates from the data.

        Fits arrival rates per incident type, month, day of the week, and hour of the day.
        Saves the results under self.lambdas and self.nye_lambdas (if fit_nye == True). Sets
        self.fitted = True when fit procedure is completed.

        Parameters
        ----------
        data: pd.DataFrame,
            The incident data.
        last_n_years: int, optional (default: 8),
            How many years to use to estimate the arrival rates. It uses the latest
            'last_n_years' years.
        fit_nye: boolean, optional (default: True),
            Whether to fit New Year's Eve separately (True) or to treat it as a regular day.
        """
        progress("Start fitting arrival rates.", verbose=self.verbose)
        # prepare data
        data = self._filter_data(data, last_n_years=last_n_years)
        data[self.day_col] = data[self.day_name_col].map(
            {"Maandag": 1, "Dinsdag": 2, "Woensdag": 3, "Donderdag": 4,
             "Vrijdag": 5, "Zaterdag": 6, "Zondag": 7})
        for col in [self.month_col, self.day_col, self.hour_col, self.month_day_col]:
            data[col] = data[col].astype(float).astype(int)

        # obtain lambdas
        progress("Obtaining lambdas..", verbose=self.verbose)
        lambdas = (data.groupby([self.type_col, self.month_col])
                       .apply(lambda x: self._get_incidents_per_hour_of_week(x, x.name[1])))

        # reindex on a complete set of types, months, and weekdays
        new_index = pd.MultiIndex.from_product(
            [data[self.type_col].unique(), np.arange(1, 13), np.arange(1, 8)],
            names=[self.type_col, self.month_col, self.day_col]
        )
        lambdas = lambdas.reindex(new_index, fill_value=0)

        # stack the hour columns and use types as columns instead
        self.lambdas = lambdas.stack().unstack(self.type_col, fill_value=0)
        progress("Lambdas obtained.", verbose=self.verbose)

        if fit_nye:
            progress("Fitting New Year's Eve.", verbose=self.verbose)
            self.nye_lambdas = self._get_incidents_at_nye(data)
            progress("New Year's Eve arrival rates fitted.", verbose=self.verbose)

        progress("Fit completed.", verbose=self.verbose)
        self.fitted = True

    def predict(self, start, end, predict_nye=True, save=False):
        """Forecast arrival rates for a given future period and save it under 'self.forecast'.

        Parameters
        ----------
        start, end: datetime object,
            The start and end dates and times (rounded to the whole hour) for the period
            to forecast.
        predict_nye: boolean, optional (default: True),
            Whether to predict NYE with high activity like in reality (True) or ignore it
            and forecast a regular day instead (False).
        """
        assert self.fitted, "First use the 'fit' method before making predictions."

        def replace_with_other(df1, df2, match_cols, fill_cols):
            """Fill one dataframe with values from another, based on specified columns."""
            assert len(match_cols) == 3, "This function needs three columns to match on."
            for i in range(len(df2)):
                mask = ((df1[match_cols[0]] == df2[match_cols[0]].iloc[i]) &
                        (df1[match_cols[1]] == df2[match_cols[1]].iloc[i]) &
                        (df1[match_cols[2]] == df2[match_cols[2]].iloc[i]))
                df1.loc[mask, fill_cols] = df2[fill_cols].iloc[i, :].values

            return df1

        # create dataframe with requested date range
        indx = pd.date_range(start=start, end=end, freq="H")
        df = pd.DataFrame({"ds": pd.Series(indx)})
        df[self.month_col] = df["ds"].dt.month
        df[self.day_col] = df["ds"].apply(lambda x: x.isoweekday())
        df[self.month_day_col] = df["ds"].dt.day
        df[self.hour_col] = df["ds"].dt.hour

        types = self.lambdas.columns
        for type_ in types:
            df[type_] = np.nan

        lambdas = self.lambdas.copy()
        lambdas.reset_index(drop=False, inplace=True)

        # fill with the overall patterns/lambdas
        progress("Filling future DataFrame..", verbose=self.verbose)
        cols = [self.month_col, self.day_col, self.hour_col]
        df = replace_with_other(df, lambdas, cols, types)
        progress("DataFrame filled with general patterns (shape: {}).".format(df.shape))

        # fill NYEs with high activity if requested
        if predict_nye:
            progress("Filling future New Year's Eves", verbose=self.verbose)
            cols = [self.month_col, self.month_day_col, self.hour_col]
            nye = self.nye_lambdas.copy()
            nye.reset_index(drop=False, inplace=True)
            df = replace_with_other(df, nye, cols, types)

            msg = "New Year's Eve forecasts added to DataFrame (shape: {})".format(df.shape)
            progress(msg, verbose=self.verbose)

        # remove added columns
        df.drop([self.month_col, self.day_col, self.month_day_col, self.hour_col],
                axis=1, inplace=True)
        self.forecast = df
        progress("Forecast created.", verbose=self.verbose)

        if save:
            self.save_forecast()

    def _filter_data(self, data, remove_unfinished_month=True, last_n_years=5):
        """Filter out some stuff for proper analysis."""
        data[self.date_col] = pd.to_datetime(data[self.date_col], dayfirst=True)
        end = data[self.date_col].max()

        if remove_unfinished_month:
            cutoff = pd.Timestamp(year=end.year, month=end.month, day=1, hour=0)
            progress("Cutting off at {}.".format(cutoff))
            data = data[data[self.date_col] < cutoff].copy()
        else:
            cutoff = end

        if last_n_years:
            start = pd.Timestamp(
                year=(cutoff.year - last_n_years),
                month=cutoff.month,
                day=cutoff.day,
                hour=cutoff.hour
            )
            progress("Using incidents after {}.".format(start))
            data = data[data[self.date_col] >= start].copy()

        progress("Data filtered.", verbose=self.verbose)
        return data

    def _get_incidents_per_hour_of_day(self, data, day_of_week=None, month=None):
        """Get average number of incidents per time (hour) of the day.

        Parameters
        ----------
        data: pd.DataFrame,
        day_of_week: int, default: None,
            Number of the weekday in [1, 7] (starting at Monday, ending at Sunday).
            If None, averages over all days. Otherwise, calculates only for the
            requested day of the week.
        month: int, default: None,
            Month number [1, 12]. If None, averages over all months. Otherwise,
            calculates only for the requested month.
        """
        start, end = data[self.date_col].min(), data[self.date_col].max()
        date_range = pd.Series(pd.date_range(start=start, end=end, freq="H"))

        # filter date range on weekday, month, and outliers
        if day_of_week is not None:
            date_range = date_range[date_range.apply(lambda x: x.isoweekday()) == day_of_week]
        if month is not None:
            date_range = date_range[date_range.dt.month == month]
        if self.ignore_dates is not None:
            date_range = date_range[~np.in1d(date_range, self.ignore_dates)]

        indx = pd.MultiIndex.from_arrays(
            [date_range.apply(lambda x: pd.Timestamp(year=x.year, month=x.month, day=x.day)),
             date_range.apply(lambda x: x.hour)],
            names=(self.date_col, self.hour_col))

        grouped = data.groupby([self.date_col, self.hour_col])[self.id_col].count()
        reindexed = grouped.reindex(indx, fill_value=0).reset_index()
        means = pd.Series(reindexed.groupby(self.hour_col)[self.id_col].mean(),
                          name=day_of_week)

        return means

    def _get_incidents_per_hour_of_week(self, data, month=None):
        """Get the mean number of incidents per every hour in a week.

        Parameters
        ----------
        data: pd.DataFrame
        month: int, default: None,
            The month number in [1, 12]. If None, averages over all months.

        Returns
        -------
        lambdas: pd.DataFrame,
            The arrival rates in a table with a row for every day of the week and a column
            for every hour of the day."""
        grouped = pd.DataFrame({d: self._get_incidents_per_hour_of_day(data, day_of_week=d, month=month) for d in np.arange(1, 8)}).T
        grouped.index.rename(self.day_col, inplace=True)
        return grouped

    def _get_incidents_at_nye(self, data):
        """Obtain average arrivals around New Year's Eve."""
        start, end = data[self.date_col].min(), data[self.date_col].max()

        years = np.array(
            [[pd.Timestamp(year=y, month=12, day=31), pd.Timestamp(year=y+1, month=1, day=1)]
             for y in range(start.year, end.year)]).flatten()

        hours = np.array([h for h in range(0, 24)], dtype=np.int8)
        indx = pd.MultiIndex.from_product([years, hours], names=[self.date_col, self.hour_col])

        data = (data.groupby([self.type_col, self.date_col, self.hour_col])[self.id_col]
                    .count()
                    .unstack(self.type_col, fill_value=0)
                    .reindex(indx, fill_value=0)
                    .reset_index())

        data[self.month_col] = data[self.date_col].dt.month
        data[self.month_day_col] = data[self.date_col].dt.day
        rates = data.groupby([self.month_col, self.month_day_col, self.hour_col]).mean()
        return rates


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
