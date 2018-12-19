import numpy as np
import pandas as pd
import osrm

from scipy.stats import lognorm, gamma, bayes_mvs
from sklearn import linear_model
import copy

from fdsim.helpers import lonlat_to_xy, xy_to_lonlat


def prepare_data_for_response_time_analysis(incidents, deployments, stations, vehicles):
    """ Prepare data for fitting dispatch, turnout, and travel times.

    Parameters
    ----------
    incidents: pd.DataFrame
        Contains the log of incidents.
    deployments: pd.DataFrame
        Contains the log of deployments.
    stations: pd.DataFrame
        Contains information on the fire stations.
    vehicles: array-like of strings
        The vehicles to keep in the resulting data. Must correspond to entries in
        the 'voertuig_groep' column in 'incidents'.

    Returns
    -------
    The merged and preprocessed DataFrame.
    """

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

    # add longitude, latitude coordinates as well
    merged[["incident_longitude", "incident_latitude"]] = \
        merged[["incident_xcoord", "incident_ycoord"]] \
        .apply(lambda x: xy_to_lonlat(x[0], x[1]), axis=1) \
        .apply(pd.Series)

    # add x, y coordinates to station location data
    stations["station_xcoord"], stations["station_ycoord"] = \
        [list(l) for l in zip(*list(stations.apply(
         lambda x: lonlat_to_xy(x["lon"], x["lat"]), axis=1)))]

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

    # filter vehicles
    df = df[np.isin(df["voertuig_groep"], vehicles)]

    return df


def get_osrm_distance_and_duration(longlat_origin, longlat_destination,
                                   osrm_host="http://192.168.56.101:5000"):
    """ Calculate distance over the road and normal travel duration from
        one point to the other.

    Parameters
    ----------
    longlat_origin: tuple(float, float)
        coordinates of the start location in decimal longitude and latitude
        (in that order).
    longlat_destination: tuple(float, float)
        coordinates of the destination in decimal longitude and latitude.
    osrm_host: str
        The URL to the OSRM API.

    Returns
    -------
    Tuple of ('distance', 'duration') according to OSRM.
    """
    osrm.RequestConfig.host = osrm_host
    result = osrm.simple_route(longlat_origin, longlat_destination,
                               output="route",
                               geometry="wkt")[0]

    return result["distance"], result["duration"]


def add_osrm_distance_and_duration(df, osrm_host="http://192.168.56.101:5000"):
    """ Calculate distance and duration over the road from station to incident
        for every incident in the data.

    Parameters
    ----------
    df: DataFrame
        The merged data of incidents and deployments. Must contain the
        following columns: {station_longitude, station_latitude,
        incident_longitude, incident_latitude}.If not present, call
        'prepare_data_for_response_time_analysis' first.
    osrm_host: str
        The URL to the OSRM API, defaults to 'http://192.168.56.101:5000', which is the
        default if running OSRM locally.

    Returns
    -------
    The DataFrame with two added columns 'osrm_distance' (meters) and 'osrm_duration'
    (seconds).
    """
    osrm.RequestConfig.host = osrm_host
    df[["osrm_distance", "osrm_duration"]] = \
        df.apply(lambda x: get_osrm_distance_and_duration(
                    (x["station_longitude"], x["station_latitude"]),
                    (x["incident_longitude"], x["incident_latitude"])),
                 axis=1).apply(pd.Series)

    return df


def fit_simple_linear_regression(data, xcol, ycol, fit_intercept=False):
    """ Fit simple linear regression on the data.

    Parameters
    ----------
    data: DataFrame
        The data to fit a model on.
    xcol: str
        The name of the column that acts as a predictor.
    ycol: str
        The name of the column acting as the dependent variable.
    fit_intercept: boolean
        If true, also fits the intercept. If false, forces intercept
        of the resulting model to 0. NOTE: Defaults to false.

    Returns
    -------
    Parameters of fitted model, i.e., a tuple of (intercept, coefficient)
    """
    lm = linear_model.LinearRegression(fit_intercept=fit_intercept)
    lm.fit(data[xcol].values.reshape(-1, 1), data[ycol])
    return lm.intercept_, lm.coef_[0]


def fit_lognorm_rv(x, **kwargs):
    """ Fit a lognormal distribution to data and return a fitted
        random variable that can be used for sampling.

    Parameters
    ----------
    x: array-like
        The data to fit.
    **kwargs: additional arguments passed to scipy.stats.lognorm.fit()

    Returns
    -------
    The fitted scipy.stats.lognorm object.
    """
    shape, loc, scale = lognorm.fit(x, **kwargs)
    return lognorm(shape, loc, scale)


def fit_gamma_rv(x, **kwargs):
    """ Fit a Gamma distribution to data and return a fitted
        random variable that can be used for sampling.

    Parameters
    ------
    x: array-like
        The data to fit.
    **kwargs: additional arguments passed to scipy.stats.gamma.fit()

    Returns
    -------
    The fitted scipy.stats.gamma object.
    """
    shape, loc, scale = gamma.fit(x, **kwargs)
    return gamma(shape, loc, scale)


def safe_bayes_mvs(x, alpha=0.9):
    """ Bayesian confidence intervals for mean and standard deviation.

    Parameters
    ----------
    x: array-like
        The data to compute confidence intervals over
    alpha: float
        The confidence level, defaults to 0.9 (90% confidence)

    Returns
    -------
    Tuple of (lower bound mean, upper bound mean,
    lower bound std, upper bound std).
    """
    if len(x) > 1:
        mean, _, std = bayes_mvs(x, alpha=alpha)
        mean_lb = mean[1][0]
        mean_ub = mean[1][1]
        std_lb = std[1][0]
        std_ub = std[1][1]
        return mean_lb, mean_ub, std_lb, std_ub
    else:
        return 0, np.inf, 0, np.inf


def sample_size_sufficient(x, alpha=0.95, max_mean_range=30, max_std_range=25):
    """ Determines if sample size is sufficient based on Bayesian
        confidence intervals and tresholds on the maximum range.

    Parameters
    ----------
    x: array-like
        The data to evaluate.
    alpha: float in range (0,1)
        The confidence level, defaults to 0.95 (95%).
    max_mean_range: float or int
        the maximum range of the confidence interval for the mean
        to classify the sample size as sufficient.
    max_std_range: float or int
        the maximum range of the confidence interval for the standard
        deviation to classify the sample size as sufficient.

    Returns
    -------
    True if the size of the confidence intervals are within the
    specified tresholds, False otherwise.
    """
    mean_lb, mean_ub, std_lb, std_ub = safe_bayes_mvs(x, alpha=alpha)
    if (mean_ub - mean_lb < max_mean_range) & (std_ub - std_lb < max_std_range):
        return True
    else:
        False


def robust_remove_travel_time_outliers(data):
    """ Remove outliers in travel time in a robust way by looking at
        the 50% most reliable data points.

    Performs two one-way outlier detection methods. It determines tresholds
    average speed and on time per distance unit (the inverse of speed) and
    cuts the values falling off on the high side. Tresholds are computed as
    follows:

    $$limit = 75% quantile + 1.5*(75% quantile - 25% quantile)$$

    Thus only data points between the 25% and 75% quantiles are used, which
    are likely to be reliable points. This makes the method robust against
    unreliable data.

    Parameters
    ----------
    data: DataFrame
        The data to remove outliers from. Assumes the columns 'osrm_distance'
        and 'inzet_rijtijd' to be present.

    Returns
    -------
    tuple of (filtered DataFrame, minimum speed, maximum speed), where speed
    is in kilometers per hour.
    """
    # add km/h and seconds/meter as columns
    data["km_h"] = data["osrm_distance"] / data["inzet_rijtijd"] * 3.6
    data["s_m"] = data["inzet_rijtijd"] / data["osrm_distance"]

    # calculate tresholds
    speed_treshold = data[["km_h", "s_m"]].describe() \
        .apply(lambda x: x["75%"] + 1.5*(x["75%"]-x["25%"]))
    max_speed = speed_treshold.loc["km_h"]
    min_speed = 1 / speed_treshold.loc["s_m"] * 3.6

    # filter data and return
    df_filtered = data[(data["km_h"] > min_speed) & (data["km_h"] < max_speed)]
    df_filtered.drop(["km_h", "s_m"], axis=1, inplace=True)
    return df_filtered, min_speed, max_speed


def model_noise_travel_time(y, x, a, b):
    """ Fit a random variable to the residual of simple linear regression
        on the travel time.

    Parameters
    ----------
    y: array-like
        The values to predict / simulate.
    x: array-like, same shape as y
        The independent variable that partially explains y.
    a: float,
        The intercept of the linear model $y ~ a + b*x$.
    b: float
        The coefficient of the linear model $y ~ a + b*x$.

    Returns
    -------
    A Lognormally distributed random variable (scipy.stats.lognorm) fitted
    on the residual: $y - (a + bx)$.
    """
    noise_factor = (np.array(y) - a) / (b*np.array(x))
    shape, loc, scale = lognorm.fit(noise_factor, loc=np.min(noise_factor), scale=1)
    rv = lognorm(shape, loc, scale)
    return rv


def model_travel_time(data):
    """ Model the travel time as a function of the estimated travel time
        from OSRM.

    Parameters
    ----------
    data: DataFrame
        Output of 'prepare_data_for_response_time_analysis'.

    Returns
    -------
    Tuple of (intercept, coefficient, residual random variable), where
    the intercept and coefficient form a linear model predicting the
    travel time based on the OSRM estimated travle duration and the
    random variable is a scipy.stats.lognorm object explaining the residual
    after prediction. The results can be used to simulate travel times for
    arbitrary incidents.
    """

    # remove records without travel time
    data = data[~data["inzet_rijtijd"].isnull()].copy()

    # remove unreliable data points
    data, _, _ = robust_remove_travel_time_outliers(data)

    # fit linear model: osrm duration -> realized duration
    intercept, coefficient = \
        fit_simple_linear_regression(data, "osrm_duration", "inzet_rijtijd")

    # model the residuals as a lognormal random variable
    residual_rv = model_noise_travel_time(data["inzet_rijtijd"],
                                          data["osrm_duration"],
                                          intercept,
                                          coefficient)

    return intercept, coefficient, residual_rv


def model_travel_time_per_vehicle(data):
    """ Model the travel time for every vehicle type separately.

    Parameters
    ----------
    data: pd.DataFrame
        Output of 'prepare_data_for_response_time_analysis'.

    Returns
    -------
    Dictionary like:
    {'vehicle' -> {'a' -> intercept,
                   'b' -> coefficient,
                   'noise_rv' -> random variable for noise}}

    See 'model_travel_time' for details on those variables.
    """
    backup_a, backup_b, backup_rv = model_travel_time(data)

    travel_time_dict = {}
    travel_time_dict["overall"] = {"a": backup_a,
                                   "b": backup_b,
                                   "noise_rv": backup_rv}

    for v in data["voertuig_groep"].unique():

        df_vehicle = data[data["voertuig_groep"] == v]

        if len(df_vehicle) > 1000:
            a, b, rv = model_travel_time(df_vehicle)
            travel_time_dict[v] = {"a": a,
                                   "b": b,
                                   "noise_rv": rv}
        else:
            travel_time_dict[v] = {"a": backup_a,
                                   "b": backup_b,
                                   "noise_rv": backup_rv}

    return travel_time_dict


def fit_dispatch_times(data, rough_upper_bound=600):
    """ Fit a lognormal random variable to the dispatch time per
        incident type.

    Parameters
    ----------
    data: DataFrame
        Merged log of deployments and incidents. All deployments in
        the data will be used for fitting, so any filtering (e.g., on
        priority or 'volgnummer') must be done in advance.
    rough_upper_bound: int
        Number of seconds to use as a rough upper bound filter, dispatch
        times above this value are considered unrealistic/unreliable and
        are removed before fitting. DEfaults to 600 seconds (10 minutes).

    Returns
    -------
    A dictionary like {'incident type' -> 'scipy.stats.lognorm object'}.
    """

    # calculate dispatch times
    data["dispatch_time"] = \
        (pd.to_datetime(data["inzet_gealarmeerd_datumtijd"], dayfirst=True) -
         pd.to_datetime(data["dim_incident_start_datumtijd"], dayfirst=True)).dt.seconds

    # filter out unrealistic values
    data = data[data["dispatch_time"] <= rough_upper_bound].copy()

    # fit one random variable on all data for the types with too few
    # observations
    backup_rv = fit_lognorm_rv(data["dispatch_time"],
                               loc=np.min(data["dispatch_time"]),
                               scale=100)

    # fit variables per incident type (use backup rv if not enough samples)
    rv_dict = {}

    for type_ in data["dim_incident_incident_type"].unique():

        X = data[data["dim_incident_incident_type"] == type_]["dispatch_time"]
        if sample_size_sufficient(X):
            rv_dict[type_] = fit_lognorm_rv(X, loc=np.min(X), scale=100)
        else:
            rv_dict[type_] = copy.deepcopy(backup_rv)

    return rv_dict


def add_parttime_fulltime_indicator(data, station_col="inzet_kazerne_groep",
                                    volunteer_stations=None):
    """ Add a column to the data, indicating whether it is a fulltime manned
    station or a parttime (volunteer) station.

    Parameters
    ----------
    data: pd.DataFrame
        The data to add the column to.
    station_col: str, optional (default: "inzet_kazerne_groep")
        The column indicating the station responsible for the deployment.
    volunteer_stations: array-like of strings, optional,
        The station names (all uppercases) of the stations that are parttime.

    Returns
    -------
    data: pd.DataFrame
        The data with an added boolean column "fulltime".
    """
    stations = data[station_col].unique()
    is_full_time_station = {station: True for station in stations}

    if volunteer_stations is not None:
        for station in volunteer_stations:
            is_full_time_station[station] = False

    data["fulltime"] = data[station_col].apply(lambda x: is_full_time_station[x])
    return data


def fit_turnout_times(data, prios=[1, 2, 3], vehicle_types=["TS", "RV", "HV", "WO"],
                      rough_lower_bound=30, rough_upper_bound=600, stations_to_exclude=None,
                      station_col="inzet_kazerne_groep", volunteer_stations=None):
    """ Fit a lognormal random variable to the turn-out time per appointment
    (fulltime/parttime), priority, and vehicle type.

    Parameters
    ----------
    data: DataFrame
        Merged log of deployments and incidents. All deployments in
        the data will be used for fitting, so any filtering (e.g., on
        priority or 'volgnummer') must be done in advance.
    prios: array-like of int, optional (default: [1, 2, 3])
        The priority levels to fit turnout times for.
    rough_lower_bound, rough_upper_bound: int
        Number of seconds to use as a rough lower and upper bound filter, turn-out
        times outside these value are considered unrealistic/unreliable and
        are removed before fitting. Defaults to 600 seconds (10 minutes) and 30 seconds.
    stations_to_exclude: array-like of str, optional (default: None)
        Stations to remove from the data before fitting. Some stations may
        imply invalid deployments (e.g, "Regio", "Onbekend"), which could
        influence the turnout times.
    station_col: str, optional (default: "inzet_kazerne_groep")
        The column in data that holds the station responsible for the deployment.
    volunteer_stations: array-like of str, optional (default: None)
        The names of the stations that are run by volunteers. These are fitted
        separately from full time stations.

    Returns
    -------
    A dictionary like {'prio' -> {'parttime' ->
    'scipy.stats.gamma object', 'fulltime' -> 'scipy.stats.gamma object'}}.
    """

    # filter stations
    data[station_col] = data[station_col].str.upper()
    if stations_to_exclude:
        data = data[~np.isin(data[station_col], stations_to_exclude)]

    # add full time or part time indicator
    data = add_parttime_fulltime_indicator(data, station_col=station_col,
                                           volunteer_stations=volunteer_stations)

    # calculate dispatch times
    data["turnout_time"] = (pd.to_datetime(data["inzet_uitgerukt_datumtijd"], dayfirst=True) -
                            pd.to_datetime(data["inzet_gealarmeerd_datumtijd"], dayfirst=True)
                            ).dt.seconds

    # filter out unrealistic values
    data = data[(data["turnout_time"] > rough_lower_bound) & 
                (data["turnout_time"] <= rough_upper_bound)].copy()

    # fit variables per incident type (use backup rv if not enough samples)
    rv_dict = {}
    for appointment in ["fulltime", "parttime"]:
        df = data[data["fulltime"] == (appointment == "fulltime")]
        backup_rv = fit_gamma_rv(df["turnout_time"], scale=100)

        df_rv_dict = {}

        for prio in prios:
            df_prio = df[df["dim_prioriteit_prio"] == prio]
            prio_backup_rv = fit_gamma_rv(df["turnout_time"], scale=100)

            prio_rv_dict = {}
            for vtype in vehicle_types:
                X = df[df["voertuig_groep"] == vtype]["turnout_time"]
                if sample_size_sufficient(X):
                    prio_rv_dict[vtype] = fit_gamma_rv(X, scale=100)
                else:
                    prio_rv_dict[vtype] = copy.deepcopy(prio_backup_rv)

            df_rv_dict[prio] = prio_rv_dict.copy()

        rv_dict[appointment] = df_rv_dict.copy()

    return rv_dict


def fit_onscene_times(data, vehicles=["TS", "HV", "RV", "WO"], rough_lower_bound=60,
                      rough_upper_bound=24*60*60):
    """ Fit a lognormal random variable to the dispatch time per
        incident type.

    Parameters
    ----------
    data: DataFrame
        Merged log of deployments and incidents. All deployments in
        the data will be used for fitting, so any filtering (e.g., on
        priority or 'volgnummer') must be done in advance.
    vehicles: array-like of strings
        The vehicles to fit on-scene times for. Optional, defaults to
        ["TS", "HV", "RV", "WO"].
    rough_lower_bound: int
        Number of seconds to use as a rough lower bound filter, on-scene
        times below this value are considered unrealistic/unreliable and
        are removed before fitting. Defaults to 60 seconds.
    rough_upper_bound: int
        Number of seconds to use as a rough upper bound filter, on-scene
        times above this value are considered unrealistic/unreliable and
        are removed before fitting. Defaults to 24*60*60 seconds (24 hours).

    Returns
    -------
    A dictionary like
    {'incident type' -> {'vehicle type' -> 'scipy.stats.gamma object'}}.
    """
    # filter vehicles
    data = data[np.in1d(data["voertuig_groep"], vehicles)].copy()

    # calculate on-scene times
    data["inzet_duration"] = \
        (pd.to_datetime(data["inzet_eind_inzet_datumtijd"], dayfirst=True) -
         pd.to_datetime(data["inzet_start_inzet_datumtijd"], dayfirst=True)).dt.seconds

    # filter out unrealistically small and large values
    data = data[(data["inzet_duration"] > rough_lower_bound) &
                (data["inzet_duration"] < rough_upper_bound)].copy()

    # fit variables per incident type (use backup rv if not enough samples)
    overall_backup_rv = fit_gamma_rv(data["inzet_duration"],
                                     floc=np.min(data["inzet_duration"])-1,
                                     scale=100)
    types = data["dim_incident_incident_type"].unique()
    rv_dict = {}

    for type_ in types:

        # backup random variable per station
        df_type = data[data["dim_incident_incident_type"] == type_]

        if sample_size_sufficient(df_type["inzet_duration"], max_mean_range=20*60,
                                  max_std_range=20*60):
            type_backup_rv = fit_gamma_rv(df_type["inzet_duration"],
                                          floc=np.min(df_type["inzet_duration"]) - 1,
                                          scale=100)
        else:
            type_backup_rv = copy.deepcopy(overall_backup_rv)

        # create dict with entry for every type with vehicle-specific data
        type_rv_dict = {}

        for v in vehicles:
            X = df_type[df_type["voertuig_groep"] == v]["inzet_duration"]
            if sample_size_sufficient(X, max_mean_range=20*60, max_std_range=20*60):
                type_rv_dict[v] = fit_gamma_rv(X,
                                               floc=np.min(X) - 1,
                                               scale=100)
            else:
                type_rv_dict[v] = copy.deepcopy(type_backup_rv)

        rv_dict[type_] = type_rv_dict.copy()

    return rv_dict


def get_coordinates_locations_stations(data, location_col="hub_vak_bk"):
    """ Obtain the coordinates of the demand locations and stations.

    Parameters
    ----------
    data: pd.DataFrame
        Merged and preprocessed data (result from 'prepare_data_for_response_time_analysis').
    location_col: str, column name of data
        The column to use as identifier for the (demand) location.
    custom_station_locations: array-like of strings
        Identifiers of the locations (in data[location_col]) of the stations in case the
        custom station locations should be used. If provided, does not use the stations
        in the data.

    Notes
    -----
    Assumes data has the following columns for coordinates: 'incident_longitude',
    'incident_latitude', 'station_longitude', 'station_latitude'. Data must be in the
    desired coordinate system already.

    Returns
    -------
    Two dictionaries. The first contains demand location coordinates:
    {'location id' -> (longitude, latitude)}
    The second holds the coordinates of the stations:
    {'station name' -> (longitude, latitude)}
    In case of custom station locations, the station name is replaced with an arbitraty
    identifier.
    """
    location_coords = (data.groupby(location_col).apply(
                       lambda x: tuple([x["incident_longitude"].mean(),
                                       x["incident_latitude"].mean()]))
                       .to_dict())

    station_coords = (data.groupby("inzet_kazerne_groep")
                      .apply(lambda x: tuple([x["station_longitude"].iloc[0],
                                             x["station_latitude"].iloc[0]]))
                      .to_dict())

    return location_coords, station_coords
