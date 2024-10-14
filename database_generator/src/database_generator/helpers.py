import os
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
import json

# Get the logger for this module
import logging
from database_generator.logging_configuration import setup_logging_for_this_script

setup_logging_for_this_script()
# Get the logger for this module
logger = logging.getLogger(__name__)


def get_config_path():
    # Check if the environment variable is set
    env_path = os.getenv("PATH_TO_THE_CONFIGURATION_FILE")

    if env_path:
        return env_path
    else:
        logger.error(
            "Configuration file path must be provided"
            " either as an environment variable 'PATH_TO_THE_CONFIGURATION_FILE'"
        )
        raise ValueError(
            "Configuration file path must be provided"
            " either as an environment variable 'PATH_TO_THE_CONFIGURATION_FILE'"
        )


# Parse the timestamps with error handling
def parse_timestamp(timestamp_str: str) -> datetime:
    try:
        # Replace '-00' with '+00:00' to make it a valid ISO 8601 format
        if timestamp_str.endswith("-00"):
            timestamp_str = timestamp_str.replace("-00", "+00:00")
        return datetime.fromisoformat(timestamp_str)
    except ValueError as e:
        logger.error(
            f"Error parsing timestamp: {timestamp_str}. {e}"
            "datetime should be in ISO 8601 format"
        )
        raise


def load_and_process_params(file_path: str) -> dict:
    """
    Load and process parameters from a JSON file, returning a flat dictionary with all parameter names and values.

    Parameters:
    ----------
    file_path : str
        The path to the JSON file containing the parameters.

    Returns:
    -------
    dict
        A dictionary containing all parameters with their names as keys and their respective values.
    """

    def recursive_extract(params: dict, result: dict = None) -> dict:
        """
        Recursively extract all key-value pairs from a nested dictionary without parent keys.

        Parameters:
        ----------
        params : dict
            The dictionary to extract key-value pairs from.
        result : dict
            The dictionary to store the flattened parameters.

        Returns:
        -------
        dict
            A dictionary containing all flattened parameters without parent keys.
        """
        if result is None:
            result = {}

        for key, value in params.items():
            if isinstance(value, dict):
                # Recursively process nested dictionaries without adding parent keys
                recursive_extract(value, result)
            else:
                # Use the current key without any parent key
                result[key] = value
                logger.info(f"{key} = {value}")

        return result

    # Load parameters from JSON file
    with open(file_path, "r") as file:
        params = json.load(file)

    # Recursively extract all parameters
    processed_params = recursive_extract(params)

    return processed_params


def validate_and_convert_datetime(dt: datetime) -> datetime:
    """
    Validates if the input is a datetime object with a UTC timezone. If not, converts it to the correct format.

    Parameters:
    -----------
    dt : datetime
        The input datetime object to validate and convert.

    Returns:
    --------
    datetime
        A timezone-aware datetime object in UTC.
    """
    # Check if the input is already a pandas Timestamp or Python datetime in UTC
    if isinstance(dt, pd.Timestamp):
        if dt.tz is None:
            logger.warning(f"Datetime '{dt}' is not timezone-aware. Converting to UTC.")
            dt = dt.tz_localize("UTC")
        elif dt.tzname() != "UTC":
            logger.warning(f"Datetime '{dt}' is not in UTC. Converting to UTC.")
            dt = dt.tz_convert("UTC")
        else:
            # Correct format and timezone; no need to convert
            logger.info(f"Datetime '{dt}' is already in the correct format.")
            return dt
    elif isinstance(dt, datetime):
        if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
            logger.warning(f"Datetime '{dt}' is not timezone-aware. Converting to UTC.")
            dt = dt.replace(tzinfo=datetime.timezone.utc)
        elif dt.tzinfo != datetime.timezone.utc:
            logger.warning(f"Datetime '{dt}' is not in UTC. Converting to UTC.")
            dt = dt.astimezone(datetime.timezone.utc)
        else:
            # Correct format and timezone; no need to convert
            logger.info(f"Datetime '{dt}' is already in the correct format.")
            return dt
    else:
        logger.warning(
            f"Input '{dt}' is not a recognized datetime format. Converting to UTC."
        )
        dt = pd.to_datetime(dt, utc=True)

    return dt


def append_and_concatenate_dataframes(
    dataframes: list[pd.DataFrame], method: str = "first"
) -> pd.DataFrame:
    """
    Appends multiple DataFrames to a list and concatenates them into a single DataFrame.
    Handles duplicate datetime indices by aggregating values based on the specified method.

    Parameters:
    -----------
    dataframes : List[pd.DataFrame]
        A list of DataFrames to be concatenated.
    method : str, optional
        The method to handle duplicate datetime indices. Options include 'mean', 'sum', 'first'.
        Defaults to 'mean'.

    Returns:
    --------
    pd.DataFrame
        A single concatenated DataFrame with duplicate datetime indices handled.
    """
    # Concatenate all DataFrames in the list
    concatenated_df = pd.concat(dataframes)

    # Handle duplicate datetime indices based on the specified method
    if method == "mean":
        concatenated_df = concatenated_df.groupby(concatenated_df.index).mean()
    elif method == "sum":
        concatenated_df = concatenated_df.groupby(concatenated_df.index).sum()
    elif method == "first":
        concatenated_df = concatenated_df.groupby(concatenated_df.index).first()
    else:
        logger.warning("Invalid method provided. Use 'mean', 'sum', or 'first'.")
        raise ValueError("Invalid method provided. Use 'mean', 'sum', or 'first'.")

    return concatenated_df


def validate_datetime_order(start_datetime: datetime, end_datetime: datetime):
    """
    Validates that the end_datetime is after the start_datetime.

    Parameters:
    -----------
    start_datetime : datetime
        The start datetime of the time series.
    end_datetime : datetime
        The end datetime of the time series.

    Raises:
    -------
    ValueError
        If end_datetime is not after start_datetime.
    """
    if end_datetime <= start_datetime:
        logger.error(
            f"end_datetime ({end_datetime}) must be after start_datetime ({start_datetime})."
        )
        raise ValueError(
            f"end_datetime ({end_datetime}) must be after start_datetime ({start_datetime})."
        )


def validate_time_range(start_datetime: pd.Timestamp, end_datetime: pd.Timestamp, frequency: str):
    """
    Validates that the time range between start_datetime and end_datetime
    is greater than or equal to the provided frequency.

    Parameters:
    -----------
    start_datetime : pd.Timestamp
        The start of the time range.
    end_datetime : pd.Timestamp
        The end of the time range.
    frequency : str
        The frequency at which data should be generated (e.g., '30s', '1min').

    Raises:
    -------
    ValueError
        If the time range is smaller than the provided frequency.
    """
    # Calculate the time difference between start_datetime and end_datetime
    time_difference = end_datetime - start_datetime
    frequency_timedelta = pd.to_timedelta(frequency)

    # Check if the time difference is smaller than the frequency
    if time_difference < frequency_timedelta:
        error_message = (
            f"Time range is too short: {time_difference} is smaller than the frequency {frequency_timedelta}. "
            "Start datetime must be before end datetime by at least the frequency."
        )
        # Log the error
        logger.error(error_message)

        # Raise a ValueError with the same message
        raise ValueError(error_message)
