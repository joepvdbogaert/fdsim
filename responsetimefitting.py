import numpy as np
import pandas as pd
import osrm

from scipy.stats import lognorm, gamma

from sklearn import linear_model

from utils import lonlat_to_xy, pre_process_station_name, xy_to_lonlat


def prepare_data_for_response_time_analysis(incidents, deployments, stations):
    """ Prepare data for fitting travel times """
    
    # specify duplicate column names before merging
    incidents.rename({"kazerne_groep":"incident_kazerne_groep"}, axis="columns", inplace=True)
    deployments.rename({"kazerne_groep":"inzet_kazerne_groep"}, axis="columns", inplace=True)

    # merge incidents and deployments
    merged = pd.merge(deployments, incidents, how="inner",
        left_on="hub_incident_id", right_on="dim_incident_id")

    # preprocess station name
    merged['inzet_kazerne_naam'] = merged['inzet_kazerne_naam'].apply(
        lambda x: pre_process_station_name(x))
    
    # rename x, y coordinate columns to avoid confusion
    merged.rename({"st_x":"incident_xcoord", "st_y":"incident_ycoord"}, 
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
    stations["kazerne"] = stations["kazerne"].apply(
        lambda x: pre_process_station_name(x))

    # and rename to avoid confusion
    stations.rename({"lon":"station_longitude", "lat":"station_latitude"},
                    axis="columns", inplace=True)

    # create the merged dataset
    df = pd.merge(merged, station_locations, left_on="inzet_kazerne_naam",
                  right_on="kazerne", how = "inner")

    # filter on priority == 1 and volgnummer == 1
    df = df[(df["dim_prioriteit_prio"]==1) & (df["inzet_terplaatse_volgnummer"]==1)]

    return df


def get_osrm_distance_and_duration(longlat_origin, longlat_destination):
    """ Calculate distance over the road and normal travel duration from 
        one point to the other.

    params
    ------
    longlat_origin: tuple(float, float)
        coordinates of the start location in decimal longitude and latitude
        (in that order).
    longlat_destination: tuple(float, float)
        coordinates of the destination in decimal longitude and latitude.

    return
    ------
    Tuple of ('distance', 'duration') according to OSRM.
    """
    result = osrm.simple_route(longlat_origin, longlat_destination,
                               output="route",
                               geometry="wkt")[0]

    return result["distance"], result["duration"]


def add_osrm_distance_and_duration(df, osrm_host="http://192.168.56.101:5000"):
    """ Calculate distance and duration over the road from station to incident
        for every incident in the data.

    params
    ------
    df: DataFrame
        merged data of incidents and deployments. Must contain the following columns: 
        {station_longitude, station_latitude, incident_longitude, incident_latitude}.
        If not present, call '..' first.
    osrm_host: str
        url to the OSRM API, defaults to 'http://192.168.56.101:5000', which is the
        default if running OSRM locally.

    return
    ------
    the DataFrame with two added columns 'osrm_distance' (meters) and 'osrm_duration'
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

    params
    ------
    data: DataFrame
        the data to fit a model on.
    xcol: str
        the name of the column that acts as a predictor.
    ycol: str
        the name of the column acting as the dependent variable.

    return
    ------
    parameters of fitter model, i.e., a tuple of (intercept, coefficient)
    """
    lm = linear_model.LinearRegression(fit_intercept=fit_intercept)
    lm.fit(data[xcol].values.reshape(-1,1), data[ycol])
    return lm.intercept_, lm.coef_[0]


def fit_lognorm_rv(x, **kwargs):
    """ Fit a lognormal distribution to data and return a fitted
        random variable that can be used for sampling.

    params
    ------
    x: array-like
        The data to fit.
    **kwargs: additional arguments passed to scipy.stats.lognorm.fit()

    return
    ------
    The fitted scipy.stats.lognorm object.
    """
    shape, loc, scale = lognorm.fit(X, kwargs)
    return lognorm(shape, loc, scale)


def fit_gamma_rv(x, **kwargs):
    """ Fit a Gamma distribution to data and return a fitted
        random variable that can be used for sampling.

    params
    ------
    x: array-like
        The data to fit.
    **kwargs: additional arguments passed to scipy.stats.gamma.fit()

    return
    ------
    The fitted scipy.stats.gamma object.
    """
    shape, loc, scale = gamma.fit(X, kwargs)
    return gamma(shape, loc, scale)


def safe_bayes_mvs(x, alpha=0.9):
    """ Bayesian confidence intervals for mean and standard deviation.

    params
    ------
    x: array-like
        the data to compute confidence intervals over
    alpha: float
        the confidence level, defaults to 0.9 (90% confidence)
    
    return
    ------
    tuple of (lower bound mean, upper bound mean, 
    lower bound std, upper bound std).
    """
    if len(x)>1:
        mean, _, std = bayes_mvs(x, alpha=alpha)
        mean_lb = mean[1][0]
        mean_ub = mean[1][1]
        std_lb = std[1][0]
        std_ub = std[1][1]
        return mean_lb, mean_ub, std_lb, std_ub 
    else: 
        return 0, np.inf, 0, np.inf


def sample_size_sufficient(x, max_mean_range=30, max_std_range=25):
    """ Determines if sample size is sufficient based on Bayesian
        confidence intervals and tresholds on the maximum range.

    params
    ------
    x: array-like
        the data to evaluate
    max_mean_range: float or int
        the maximum range of the confidence interval for the mean
        to classify the sample size as sufficient.
    max_std_range: float or int
        the maximum range of the confidence interval for the standard
        deviation to classify the sample size as sufficient.

    return
    ------
    True if the size of the confidence intervals are within the
    specified tresholds, False otherwise.
    """
    mean_lb, mean_ub, std_lb, std_ub = safe_bayes_mvs(x, alpha=alpha)
    if (mean_ub-mean_ls < max_mean_range) & (std_ub-std_lb < max_std_range):
        return True
    else:
        False


def robust_remove_travel_time_outliers(data):
    """ Remove outliers in travel time in a robust way by looking at 
        the 50% most reliable data points.

    params
    ------
    data: DataFrame
        The data to remove outliers from. Assumes the columns 'osrm_distance'
        and 'inzet_rijtijd' to be present.
    
    notes
    -----
    Performs two one-way outlier detection methods. It determines tresholds 
    average speed and on time per distance unit (the inverse of speed) and 
    cuts the values falling off on the high side. Tresholds are computed as
    follows:

    $$limit = 75% quantile + 1.5*(75% quantile - 25% quantile)$$

    Thus only data points between the 25% and 75% quantiles are used, which
    are likely to be reliable points. This makes the method robust against
    unreliable data.

    return
    ------
    tuple of (filtered DataFrame, minimum speed, maximum speed), where speed 
    is in kilometers per hour.
    """
    # add km/h and seconds/meter as columns
    data["km_h"] = data["osrm_distance"] / data["inzet_rijtijd"] * 3.6
    data["s_m"] = data["inzet_rijtijd"] / data["osrm_distance"]
    
    # calculate tresholds
    speed_treshold = data[["km_h", "s_m"]] \
                        .describe() \
                        .apply(lambda x: x["75%"] + 1.5*(x["75%"]-x["25%"]))
    max_speed = speed_treshold.loc["km_h"]
    min_speed = 1 / speed_treshold.loc["s_m"] * 3.6
    
    # filter data and return
    df_filtered = data[(data["km_h"] > min_speed) & (data["km_h"] < max_speed)]
    df_filtered.drop(["km_h", "s_m"], axis=1, inplace=True)
    return df_filtered, min_speed, max_speed


def model_residual_travel_time(y, x, a, b):
    """ Fit a random variable to the residual of simple linear regression
        on the travel time.

    params
    ------
    y: array-like
        The values to predict / simulate.
    x: array-like, same shape as y
        The independent variable that partially explains y.
    a: float,
        The intercept of the linear model $y ~ a + b*x$.
    b: float
        The coefficient of the linear model $y ~ a + b*x$.

    return
    ------
    A Lognormally distributed random variable (scipy.stats.lognorm) fitted 
    on the residual: $y - (a + bx)$.
    """
    residual = np.array(y) - (a + b*np.array(x))
    shape, loc, scale = lognorm.fit(residual, loc=np.min(residual), scale=1)
    rv = lognorm(shape, loc, scale)
    return rv


def model_travel_time(incidents, deployments, stations):
    """ Model the travel time as a function of the estimated travel time 
        from OSRM.

    params
    ------
    incidents: DataFrame
        Incident log.
    deployments: DataFrame
        Deployment log.
    stations: DataFrame
        Station location data.

    return
    ------
    Tuple of (intercept, coefficient, residual random variable), where
    the intercept and coefficient form a linear model predicting the 
    travel time based on the OSRM estimated travle duration and the 
    random variable is a scipy.stats.lognorm object explaining the residual
    after prediction. The results can be used to simulate travel times for
    arbitrary incidents.
    """
    # prepare and merge data
    data = prepare_data_for_travel_time_analysis(
        incidents, deployments, stations)

    # remove records without travel time
    data = data[~data["inzet_rijtijd"].isnull()]

    # remove unreliable data points
    data, _, _ = robust_remove_travel_time_outliers(data)

    # fit linear model: osrm duration -> realized duration
    intercept, coefficient = \
        fit_simple_linear_regression(data, "osrm_duration", "inzet_rijtijd")

    # model the residuals as a lognormal random variable
    residual_rv = model_residual_travel_time(data["inzet_rijtijd"],
                                             data["osrm_duration"],
                                             intercept,
                                             coefficient)

    return intercept, coefficient, residual_rv


def fit_dispatch_times(data, rough_upper_bound=600):
    """ Fit a lognormal random variable to the dispatch time per
        incident type.

    params
    ------
    data: DataFrame
        Merged log of deployments and incidents. All deployments in 
        the data will be used for fitting, so any filtering (e.g., on 
        priority or 'volgnummer') must be done in advance.
    rough_upper_bound: int
        Number of seconds to use as a rough upper bound filter, dispatch
        times above this value are considered unrealistic/unreliable and
        are removed before fitting. DEfaults to 600 seconds (10 minutes).

    return  
    ------
    A dictionary like {'incident type' -> 'scipy.stats.lognorm object'}.
    """

    # calculate dispatch times
    data["dispatch_time"] = (pd.to_datetime(data["inzet_gealarmeerd_datumtijd"])
        - pd.to_datetime(data["dim_incident_start_datumtijd"])).dt.seconds

    # filter out unrealistic values
    data = data[data["dispatch_time"] <= rough_upper_bound]

    # fit one random variable on all data for the types with too few
    # observations
    backup_rv = fit_lognorm_rv(data["dispatch_time"],
                               loc=np.min(data["dispatch_time"]),
                               scale=100)

    # fit variables per incident type (use backup rv if not enough samples)
    rv_dict = {}
    for type_ in data["dim_incident_incident_type"].unique():
        X = data[data["dim_incident_incident_type"]==type_]["dispatch_time"]
        if sample_size_sufficient(X):
            rv_dict[type_] = fit_lognorm_rv(X, loc=np.min(X), scale=100)
        else:
            rv_dict[type_] = backup_rv.copy()

    return rv_dict


def fit_turnout_times(data, station_names, rough_upper_bound=600):
    """ Fit a lognormal random variable to the dispatch time per
        incident type.

    params
    ------
    data: DataFrame
        Merged log of deployments and incidents. All deployments in 
        the data will be used for fitting, so any filtering (e.g., on 
        priority or 'volgnummer') must be done in advance.
    station_names: array-like of strings
        Names of the stations to fit turnout times for. Must match the
        names in data["inzet_kazerne_groep"]
    rough_upper_bound: int
        Number of seconds to use as a rough upper bound filter, turn-out
        times above this value are considered unrealistic/unreliable and
        are removed before fitting. DEfaults to 600 seconds (10 minutes).

    return  
    ------
    A dictionary like {'station' -> {'incident type' -> 
    'scipy.stats.gamma object'}}.
    """
    # filter stations
    station_names = pd.Series(station_names).str.upper()
    data["inzet_kazerne_groep"] = data["inzet_kazerne_groep"].str.upper()
    data = data[np.isin(data["inzet_kazerne_groep"], station_names)]

    # calculate dispatch times
    df["turnout_time"] = (pd.to_datetime(df["inzet_uitgerukt_datumtijd"]) - 
                          pd.to_datetime(df["inzet_gealarmeerd_datumtijd"])
                          ).dt.seconds

    # filter out unrealistic values
    data = data[data["turnout_time"] <= rough_upper_bound]

    # fit one random variable on all data for the types with too few
    # observations
    overall_backup_rv = fit_gamma_rv(data["turnout_time"],
                                     floc=np.min(data["turnout_time"])-0.1,
                                     scale=100)

    # fit variables per incident type (use backup rv if not enough samples)
    types = data["dim_incident_incident_type"].unique()
    rv_dict = {}

    for station in station_names:

        # backup rv per station
        dfstation = data[data["inzet_kazerne_groep"]==station]
        station_backup_rv = fit_gamma_rv(dfstation["turnout_time"],
                                         floc=np.min(X)-0.1, scale=100)

        # create dict with entry for every type with station-specific data
        station_rv_dict = {}
        for type_ in types:
            X = data[data["dim_incident_incident_type"]==type_]["turnout_time"]
            if sample_size_sufficient(X):
                station_rv_dict[type_] = fit_gamma_rv(X,
                                                      floc=np.min(X)-0.1,
                                                      scale=100)
            else:
                station_rv_dict[type_] = station_backup_rv.copy()

        rv_dict[station] = station_rv_dict.copy()

    return rv_dict