import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from database_generator.helpers import (
    validate_and_convert_datetime,
    validate_datetime_order,
)

# Set up logging
from database_generator.logging_configuration import setup_logging_for_this_script

setup_logging_for_this_script()
logger = logging.getLogger(__name__)


def generate_stable_toy_data(
    start_datetime: datetime, end_datetime: datetime, seed_for_random: int = None
) -> pd.DataFrame:
    """
    Generate a stable toy dataset for an industrial pump system with correlated variables.

    Parameters:
    -----------
    start_datetime : datetime
        The start datetime of the time series.
    end_datetime : datetime
        The end datetime of the time series.
    seed_for_random : int, optional
        Seed for random number generation for reproducibility. Defaults to None.

    Returns:
    --------
    pd.DataFrame
        A DataFrame containing the generated time series data with a 'Timestamp' index.
    """
    # Validate and convert datetime inputs
    start_datetime = validate_and_convert_datetime(start_datetime)
    end_datetime = validate_and_convert_datetime(end_datetime)

    # Validate that end_datetime is after start_datetime
    validate_datetime_order(start_datetime, end_datetime)

    # Set the seed for reproducibility
    if seed_for_random is not None:
        np.random.seed(seed_for_random)

    # Generate a date range
    date_range = pd.date_range(
        start=start_datetime, end=end_datetime, freq="30s", tz="UTC"
    )
    number_of_rows = len(date_range)

    # Generate base data with correlations
    temperature = np.random.normal(loc=75, scale=1, size=number_of_rows)
    pressure = (
        3
        - 0.01 * (temperature - 75)
        + np.random.normal(loc=0, scale=0.05, size=number_of_rows)
    )
    flow_rate = (
        300
        + 10 * (3 - pressure)
        + np.random.normal(loc=0, scale=5, size=number_of_rows)
    )
    vibration = 0.1 * np.sqrt(flow_rate * pressure) + np.random.normal(
        loc=0, scale=0.05, size=number_of_rows
    )
    humidity = np.random.normal(loc=40, scale=5, size=number_of_rows)

    # Create a DataFrame
    df = pd.DataFrame(
        {
            "Timestamp": date_range,
            "Temperature_C": temperature,
            "Pressure_MPa": pressure,
            "Vibration_mm_s": vibration,
            "Flow_Rate_l_min": flow_rate,
            "Humidity_%": humidity,
        }
    )

    # Set Timestamp as the index
    df.set_index("Timestamp", inplace=True)

    return df


def introduce_exponential_anomalies(
    start_datetime: datetime,
    end_datetime: datetime,
    variable: str,
    increase_rate: float = 0.01,
    seed_for_random: int = None,
) -> pd.DataFrame:
    """
    Generates a stable toy dataset and introduces an exponential anomaly by gradually increasing the specified variable
    over a given time period.

    Parameters:
    -----------
    start_datetime : datetime
        The start datetime of the time series.
    end_datetime : datetime
        The end datetime of the time series.
    variable : str
        The column name of the variable to introduce the anomaly.
    increase_rate : float, optional
        The rate at which the variable increases exponentially over time. Default is 0.01.
    seed_for_random : int, optional
        Seed for random number generation for reproducibility. Defaults to None.

    Returns:
    --------
    pd.DataFrame
        A DataFrame with the simulated anomaly.
    """
    # Validate and convert datetime inputs
    start_datetime = validate_and_convert_datetime(start_datetime)
    end_datetime = validate_and_convert_datetime(end_datetime)

    # Validate that end_datetime is after start_datetime
    validate_datetime_order(start_datetime, end_datetime)

    # Generate stable toy data
    df_stable = generate_stable_toy_data(start_datetime, end_datetime, seed_for_random)

    # Make a copy of the original DataFrame to avoid modifying it directly
    df_copy = df_stable.copy()

    # Generate a mask for the time period where the anomaly should occur
    mask = (df_copy.index >= start_datetime) & (df_copy.index <= end_datetime)

    # Debug: Print the number of affected rows
    num_affected = mask.sum()
    logger.info(f"Number of rows affected by the anomaly: {num_affected}")

    if num_affected == 0:
        logger.warning(
            "No rows found in the specified time range. Check start_datetime and end_datetime."
        )
        return df_copy

    # Apply an exponential increase to the specified variable within the specified time range
    df_copy.loc[mask, variable] += (
        np.exp(np.linspace(0, increase_rate * num_affected, num_affected)) - 1
    )

    return df_copy


def simulate_broken_sensor(
    variable: str,
    start_datetime: datetime,
    end_datetime: datetime,
    seed_for_random: int = None,
    mode: str = "stuck",
    value: float = None,
) -> pd.DataFrame:
    """
    Simulates a broken sensor by introducing anomalies into the specified variable.

    Parameters:
    -----------
    variable : str
        The column name of the variable to simulate the sensor failure on.
    start_datetime : datetime
        The start datetime of the time series.
    end_datetime : datetime
        The end datetime of the time series.
    seed_for_random : int, optional
        Seed for random number generation for reproducibility. Defaults to None.
    mode : str, optional
        The type of sensor failure. Options include 'stuck', 'jump', 'spike', and 'dropout'. Default is 'stuck'.
    value : float, optional
        The value to use for 'stuck' or 'jump' modes. Defaults to None, in which case the function determines a value.

    Returns:
    --------
    pd.DataFrame
        A DataFrame with the simulated sensor failure.
    """
    # Validate and convert datetime inputs
    start_datetime = validate_and_convert_datetime(start_datetime)
    end_datetime = validate_and_convert_datetime(end_datetime)

    # Validate that end_datetime is after start_datetime
    validate_datetime_order(start_datetime, end_datetime)

    # Generate stable toy data
    df_stable = generate_stable_toy_data(start_datetime, end_datetime, seed_for_random)

    # Make a copy of the original DataFrame to avoid modifying it directly
    df_copy = df_stable.copy()

    # Generate a mask for the time period where the anomaly should occur
    mask = (df_copy.index >= start_datetime) & (df_copy.index <= end_datetime)

    # Simulate different types of sensor failures
    if mode == "stuck":
        # Sensor gets stuck at a constant value
        if value is None:
            value = df_copy[variable][mask].mean()
        df_copy.loc[mask, variable] = value

    elif mode == "jump":
        # Sensor jumps to an unusually high or low value
        if value is None:
            value = df_copy[variable].mean() + 2 * df_copy[variable].std()
        df_copy.loc[mask, variable] += value

    elif mode == "spike":
        # Sensor produces intermittent spikes
        spike_indices = (
            df_copy.loc[mask].sample(frac=0.1).index
        )  # Select 10% of the time points within the period
        df_copy.loc[spike_indices, variable] = (
            df_copy[variable].mean() + 15 * df_copy[variable].std()
        )

    elif mode == "dropout":
        # Sensor stops reporting data (NaN values)
        df_copy.loc[mask, variable] = np.nan

    return df_copy
