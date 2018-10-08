import numpy as np
import pandas as pd


def prepare_incidents_for_spatial_analysis(incidents, location_col):
    """ Perform initial preprocessing tasks before fitting
        parameters and obtaining probabilities from the incident data.

    params
    ------
    incidents: pd.DataFrame
        The incident data to prepare.
    location_col: str
        The column in 'incidents' that identifies the demand location.

    notes
    -----
    Some tasks to perform before fitting:
        1. Remove NaNs in location
        2. Cast or load location column as int->string
        3. remove incidents outside AA
        4. ...

    returns
    -------
    The prepared DataFrame.
    """
    print("\tPreparing data for spatial analysis...")
    # 1. remove NaNs in location
    data = incidents[~incidents["hub_vak_bk"].isnull()].copy()

    # 2. cast 'vak' to string
    data["hub_vak_bk"] = data["hub_vak_bk"].astype(int).astype(str)

    # 3. only keep those in Amsterdam-Amstelland
    data = data[data["hub_vak_bk"].str[0:2]=="13"].copy()

    return data


def get_prio_probabilities_per_type(incidents):
    """ Create dictionary with the probabilities of having 
        priority 1, 2, and 3 for every incident type. 

    params
    ------
    incidents: pd.DataFrame
        Contains the log of incidents from which the probabilities
        should be obtained.

    returns
    -------
    Dictionary with incident type names as keys and lists of length 3
    as elements, where probabilities of prio 1, 2, 3 are in position
    0, 1, 2 respectively.
    """

    # filter out null values and prio 5
    incidents = incidents[~incidents["dim_prioriteit_prio"].isnull()]
    incidents = incidents[incidents["dim_prioriteit_prio"]!=5]

    prio_per_type = incidents.groupby( 
        ["dim_incident_incident_type", "dim_prioriteit_prio"]) \
        ["dim_incident_id"].count().reset_index()

    prio_per_type["prio_probability"] = prio_per_type.groupby(
        ["dim_incident_incident_type"])\
        ["dim_incident_id"].apply(lambda x: x/x.sum())

    prio_probabilities = pd.pivot_table(prio_per_type,
        columns="dim_incident_incident_type", values="prio_probability",
        index="dim_prioriteit_prio").fillna(0)

    return {col : list(prio_probabilities[col]) for col 
        in prio_probabilities.columns}


def get_vehicle_requirements_probabilities(incidents, deployments):
    """ Calculate the probabilities of needing a number of vehicles of a
        specific type for a specified incident type.

    params
    ------
    incidents: pd.DataFrame
        The log of incidetns to extract probabilities from.
    deployments: pd.DataFrame
        The log of deployments to extract probabilities from.
    
    return
    ------
    Nested dictionary like {"incident type": {"vehicles": prob}}.
    """

    # add incident type to the deployment data
    deployments = deployments.merge(
        incidents[["dim_incident_id", "dim_incident_incident_type"]],
        left_on="hub_incident_id", right_on="dim_incident_id", how="left")
    
    # filter out missing values and create tuples of needed vehicle types
    deployments = deployments[~deployments["inzet_voertuig_code"].isnull()]
    grouped = deployments.groupby(
        ["dim_incident_id", "dim_incident_incident_type"]) \
         .apply(lambda x: tuple(x["inzet_voertuig_code"].sort_values())) \
         .reset_index()
    
    # count occurences of every combination of vehicles per type
    counted = grouped.groupby(["dim_incident_incident_type", 0]) \
        ["dim_incident_id"].count().reset_index()
    counted["prob"] = counted.groupby("dim_incident_incident_type") \
        ["dim_incident_id"].transform(lambda x: x / x.sum())

    # create dictionary and return
    counted_dict = counted.groupby("dim_incident_incident_type").apply(
        lambda x: x.set_index(0)["prob"].to_dict())

    return counted_dict


def get_spatial_distribution_per_type(incidents, location_col="hub_vak_bk"):
    """ Obtain the distribution over demand locations for
        every incident type.
    
    params
    ------
    incidents: pd.DataFrame
        The log of incidents to obtain probabilities from.
    location_col: str (optional)
        The column in 'incidents' to use as identifier for
        demand location.
    
    return
    ------
    Dictionary like {"type": {"location": probability}}. 
    """
    
    # filter out missing values and other irrelevant observations
    incidents = prepare_incidents_for_spatial_analysis(incidents,
                                                       location_col)

    print("\tFinding spatial distributions per incident type...")
    # group and count
    counted = incidents.groupby(
        ["dim_incident_incident_type", location_col]) \
        ["dim_incident_id"] \
        .count() \
        .reset_index()

    counted["prob"] = counted.groupby("dim_incident_incident_type") \
        ["dim_incident_id"].transform(lambda x: x / x.sum())

    counted_dict = counted.groupby("dim_incident_incident_type") \
        .apply(lambda x: x.set_index(location_col)["prob"].to_dict())

    return counted_dict