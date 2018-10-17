import os
import warnings
from abc import abstractmethod, ABCMeta

import numpy as np
import pandas as pd
from fbprophet import Prophet

from definitions import ROOT_DIR


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


class ProphetIncidentPredictor(BaseIncidentPredictor):
    """ Class that forecasts incident rate for different incident types.

    Uses Facebook's Prophet to create a forecast of the incident rate.
    It does so by calculating the hourly arrivals per incident type, then
    treating this as a signal/time series and decomposing it into trend,
    yearly pattern, weekly pattern, and daily pattern.

    Example
    -------
    >>> predictor = IncidentPredictor(load_forecast=False)
    >>> predictor.fit(incident_data)
    >>> predictor.predict(periods=365*24, freq="H", save=True)
    >>> forecast = predictor.get_forecast()
    >>> forecast.head()

    Parameters
    ----------
    load_forecast: boolean
        Whether to load a pre-existing forecast from disk.
        Defaults to True, since recomputing forecasts is costly.
    fd_dir: str
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
        self.fc_dir = os.path.join(ROOT_DIR, fc_dir)
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

        This function first prepare the data and saves the prepared data
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

        self.incidents, self.time_index = self._prep_data_for_prediction(data)
        self.models_dict = dict()

        for type_ in self.types:
            if self.verbose:
                print("Fitting model for type {}...".format(type_))

            m = Prophet()
            dfprophet = self._create_prophet_data(self.incidents, type_,
                                                  self.time_index)
            m.fit(dfprophet)
            self.models_dict[type_] = m

        self.fitted = True
        if self.verbose:
            print("Models fitted.")

    def predict(self, periods=365*24, freq="H", save=False):
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
            Any valid frequency for pd.date_range, such as 'H', 'D', or 'M'.
        """
        assert self.fitted, \
            "First use 'fit()' to fit a model before predicting."
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
        """ Format time columns and create an hourly index needed
            for prediction.

        Parameters
        ----------
        incidents: pd.DataFrame
            The incident data to prepare.

        Returns
        -------
        The prepared DataFrame and a pd.Index with hourly timestamps.
        """
        incidents["dim_tijd_uur"] = incidents["dim_tijd_uur"].astype(float)\
            .astype(int).astype(str).str.zfill(2)
        incidents["dim_datum_datum"] = pd.to_datetime(
            incidents["dim_datum_datum"]).dt.strftime("%Y-%m-%d")
        new_index = incidents.groupby(["dim_datum_datum", "dim_tijd_uur"]
                                      )["dim_incident_id"].count().index
        return incidents, new_index

    def _create_prophet_data(self, incidents, type_, new_index):
        """ Create a DataFrame in the format required by Prophet.

        Parameters
        ----------
        incidents: pd.DataFrame
            The incident data.
        type_: str
            The incident type to make a DataFrame for.
        new_index: pandas.Index object
            Specifies the times that the resulting DataFrame should contain.

        Returns
        -------
        A DataFrame with two columns: 'ds' and 'y', where 'ds' is the timestamp
        and 'y' is the number of incidents per time_unit. This DataFrame can be
        used directly as input for Prophet.fit().
        """
        dfprophet = incidents[["dim_incident_id", "dim_datum_datum",
                               "dim_tijd_uur"]]
        dfprophet = dfprophet[dfprophet["dim_incident_incident_type"] == type_]
        dfprophet = dfprophet.groupby(["dim_datum_datum", "dim_tijd_uur"]
                                      )["dim_incident_id"].count()
        dfprophet = dfprophet.reindex(new_index, fill_value=0).reset_index()
        dfprophet.rename(columns={"dim_incident_id": "y"}, inplace=True)
        dfprophet["ds"] = pd.to_datetime(dfprophet["dim_datum_datum"] + " " +
                                         dfprophet["dim_tijd_uur"],
                                         format="%Y-%m-%d %H")

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

        Returns
        -------
        The sampling dictionary as described above.
        """
        assert self.forecast is not None, \
            ("No forecast available, instantiate with load_forecast=True "
             "or use .fit() and .predict() to create one.")

        if incident_types is not None:
            fc = self.forecast[["ds"] + list(incident_types)].copy()
        else:
            fc = self.forecast.copy()

        fc["ds"] = pd.to_datetime(fc["ds"])
        if start_time is None:
            start_time = fc["ds"].min()
        if end_time is None:
            end_time = fc["ds"].max()

        fc = fc[(fc["ds"] >= start_time) & (fc["ds"] < end_time)]
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
