import numpy as np
import pandas as pd


def prepare_incidents_for_spatial_analysis(incidents):
    """ Perform initial preprocessing tasks before fitting
        parameters and obtaining probabilities from the incident data.

    Parameters
    ----------
    incidents: pd.DataFrame
        The incident data to prepare.

    Notes
    -----
    Some tasks to perform before fitting:
        1. Remove NaNs in location and building function
        2. Cast or load location column as int->string
        3. remove incidents outside AA
        4. ...

    Returns
    -------
    The prepared DataFrame.
    """

    # 1. remove NaNs in location and in building function
    data = incidents[~incidents["hub_vak_bk"].isnull()].copy()
    data = data[~data["inc_dim_object_functie"].isnull()].copy()

    # 2. cast 'vak' to string
    data["hub_vak_bk"] = data["hub_vak_bk"].astype(int).astype(str)

    # 3. only keep those in Amsterdam-Amstelland
    data = data[data["hub_vak_bk"].str[0:2] == "13"].copy()

    return data


def get_prio_probabilities_per_type(incidents):
    """ Create dictionary with the probabilities of having
        priority 1, 2, and 3 for every incident type.

    Parameters
    ----------
    incidents: pd.DataFrame
        Contains the log of incidents from which the probabilities
        should be obtained.

    Returns
    -------
    Dictionary with incident type names as keys and lists of length 3
    as elements, where probabilities of prio 1, 2, 3 are in position
    0, 1, 2 respectively.
    """

    # filter out null values and prio 5
    incidents = incidents[~incidents["dim_prioriteit_prio"].isnull()]
    incidents = incidents[incidents["dim_prioriteit_prio"] != 5]

    grouped = incidents.groupby(["dim_incident_incident_type", "dim_prioriteit_prio"])
    prio_per_type = grouped["dim_incident_id"].count().reset_index()

    prio_per_type["prio_probability"] = prio_per_type \
        .groupby(["dim_incident_incident_type"])["dim_incident_id"] \
        .apply(lambda x: x / x.sum())

    prio_probabilities = pd.pivot_table(prio_per_type,
                                        columns="dim_incident_incident_type",
                                        values="prio_probability",
                                        index="dim_prioriteit_prio").fillna(0)

    return {col: list(prio_probabilities[col]) for col
            in prio_probabilities.columns}


def get_vehicle_requirements_probabilities(incidents, deployments, vehicles):
    """ Calculate the probabilities of needing a number of vehicles of a
        specific type for a specified incident type.

    Parameters
    ----------
    incidents: pd.DataFrame
        The log of incidetns to extract probabilities from.
    deployments: pd.DataFrame
        The log of deployments to extract probabilities from.

    Returns
    -------
    Nested dictionary like {"incident type": {"vehicles": prob}}.
    """
    deployments = deployments[np.isin(deployments["voertuig_groep"], vehicles)]
    # add incident type to the deployment data
    deployments = deployments.merge(
        incidents[["dim_incident_id", "dim_incident_incident_type"]],
        left_on="hub_incident_id", right_on="dim_incident_id", how="left")

    # filter out missing values and create tuples of needed vehicle types
    deployments = deployments[~deployments["voertuig_groep"].isnull()]
    grouped = deployments.groupby(["dim_incident_id", "dim_incident_incident_type"]) \
        .apply(lambda x: tuple(x["voertuig_groep"].sort_values())) \
        .reset_index()

    # count occurences of every combination of vehicles per type
    counted = grouped.groupby(["dim_incident_incident_type", 0])["dim_incident_id"] \
                     .count() \
                     .reset_index()

    counted["prob"] = counted.groupby("dim_incident_incident_type")["dim_incident_id"] \
                             .transform(lambda x: x / x.sum())

    # create dictionary and return
    counted_dict = counted.groupby("dim_incident_incident_type").apply(
        lambda x: x.set_index(0)["prob"].to_dict())

    return counted_dict


def get_spatial_distribution_per_type(incidents, location_col="hub_vak_bk"):
    """ Obtain the distribution over demand locations for
        every incident type.

    Parameters
    ----------
    incidents: pd.DataFrame
        The log of incidents to obtain probabilities from.
    location_col: str (optional)
        The column in 'incidents' to use as identifier for
        demand location.

    Returns
    -------
    Dictionary like {"type": {"location": probability}}.
    """

    # filter out missing values and other irrelevant observations
    incidents = prepare_incidents_for_spatial_analysis(incidents)

    # group and count
    grouped = incidents.groupby(["dim_incident_incident_type", location_col])
    counted = grouped["dim_incident_id"].count().reset_index()

    counted["prob"] = counted.groupby("dim_incident_incident_type")["dim_incident_id"] \
                             .transform(lambda x: x / x.sum())

    counted_dict = counted.groupby("dim_incident_incident_type") \
        .apply(lambda x: x.set_index(location_col)["prob"].to_dict()) \
        .to_dict()

    return counted_dict


def get_building_function_probabilities(incidents, location_col="hub_vak_bk"):
    """ Find the distribution of building functions per demand location.

    Parameters
    ----------
    incidents: pd.DataFrame
        The log of incidents to obtain building function distributions from.
    location_col: str
        The column name in 'incidents' that identifies the demand location.

    Returns
    -------
    A nested dictionary like:
    {'location id' -> {'incident type' -> {'building function' -> probability}}}.
    """

    incidents = prepare_incidents_for_spatial_analysis(incidents)

    grouped = incidents.groupby([location_col, "dim_incident_incident_type",
                                 "inc_dim_object_functie"])["dim_incident_id"] \
                       .count().reset_index()

    grouped["prob"] = (
        grouped.groupby([location_col, "dim_incident_incident_type"])["dim_incident_id"]
               .transform(lambda x: x / x.sum()))

    partial_dict = grouped.groupby([location_col, "dim_incident_incident_type"]).apply(
        lambda x: x.set_index("inc_dim_object_functie")["prob"].to_dict()) \
        .reset_index()

    building_dict = partial_dict.groupby(location_col) \
        .apply(lambda x: x.set_index("dim_incident_incident_type")[0].to_dict()) \
        .to_dict()

    return building_dict
