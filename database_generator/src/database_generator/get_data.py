import numpy as np
import pandas as pd
import datetime as datetime


# Get the logger for this module
import logging
from database_generator.logging_configuration import setup_logging_for_this_script
setup_logging_for_this_script()
# Get the logger for this module
logger = logging.getLogger(__name__)


# def generate_stable_toy_data(start_date: str, end_date: str, seed_for_random: int = None) -> pd.DataFrame:
#     """
#     Generate a stable toy dataset for an industrial pump system with correlated variables.

#     This function generates time series data for variables such as Temperature, Pressure, Flow Rate, Vibration, and Humidity
#     in an industrial pump system. The data is generated between the provided start and end dates, with a 5-minute frequency.

#     Parameters:
#     -----------
#     start_date : str
#         The start date of the time series in ISO format (e.g., '2024-01-01 00:00:00').
#     end_date : str
#         The end date of the time series in ISO format (e.g., '2024-01-02 00:00:00').
#     seed_for_random : int, optional
#         Seed for random number generation for reproducibility. Defaults to None.

#     Returns:
#     --------
#     pd.DataFrame
#         A DataFrame containing the generated time series data with a 'Timestamp' index.
#     """
#     # Set the seed for reproducibility
#     if seed_for_random is not None:
#         np.random.seed(seed_for_random)
    
#     # Generate a date range
#     date_range = pd.date_range(start=start_date, end=end_date, freq='30s', tz='UTC')
#     number_of_rows = len(date_range)
    
#     # Generate base data with correlations
#     # Temperature: Normally distributed around 75°C with small fluctuations
#     temperature = np.random.normal(loc=75, scale=1, size=number_of_rows)
    
#     # Pressure: Correlated with temperature, slightly decreasing with higher temperatures
#     pressure = 3 - 0.01 * (temperature - 75) + np.random.normal(loc=0, scale=0.05, size=number_of_rows)
    
#     # Flow Rate: Generally stable, slightly increasing with lower pressure (inverse correlation)
#     flow_rate = 300 + 10 * (3 - pressure) + np.random.normal(loc=0, scale=5, size=number_of_rows)
    
#     # Vibration: Non-linear increase with flow_rate and pressure
#     vibration = 0.1 * np.sqrt(flow_rate * pressure) + np.random.normal(loc=0, scale=0.05, size=number_of_rows)
    
#     # Humidity: Independent of the other variables, normal fluctuations
#     humidity = np.random.normal(loc=40, scale=5, size=number_of_rows)
    
#     # Create a DataFrame
#     df = pd.DataFrame({
#         'Timestamp': date_range,
#         'Temperature_C': temperature,
#         'Pressure_MPa': pressure,
#         'Vibration_mm_s': vibration,
#         'Flow_Rate_l_min': flow_rate,
#         'Humidity_%': humidity
#     })
    
#     # Set Timestamp as the index
#     df.set_index('Timestamp', inplace=True)
    
#     return df


# def introduce_exponential_anomalies(
#     start_date: str,
#     end_date: str,
#     variable: str,
#     start_time: datetime,
#     end_time: datetime,
#     increase_rate: float = 0.01,
#     seed_for_random: int = None
# ) -> pd.DataFrame:
#     """
#     Generates a stable toy dataset and introduces an exponential anomaly by gradually increasing the specified variable
#     over a given time period.

#     Parameters:
#     -----------
#     start_date : str
#         The start date of the time series in ISO format (e.g., '2024-01-01 00:00:00').
#     end_date : str
#         The end date of the time series in ISO format (e.g., '2024-01-02 00:00:00').
#     variable (str): The column name of the variable to introduce the anomaly.
#     start_time (datetime): Start time of the anomaly.
#     end_time (datetime): End time of the anomaly.
#     increase_rate (float): The rate at which the variable increases exponentially over time. Default is 0.01.
#     seed_for_random : int, optional
#         Seed for random number generation for reproducibility. Defaults to None.

#     Returns:
#     --------
#     pd.DataFrame: A new DataFrame with the simulated anomaly.
#     """
#     # Generate stable toy data
#     df_stable = generate_stable_toy_data(start_date, end_date, seed_for_random)

#     # Make a copy of the original DataFrame to avoid modifying it directly
#     df_copy = df_stable.copy()

#     # Check if the index is already in ISO datetime format
#     if not pd.api.types.is_datetime64_any_dtype(df_copy.index):
#         logger.warning("Index is not in datetime format. Converting to datetime.")
#         df_copy.index = pd.to_datetime(df_copy.index)

#     # Generate a mask for the time period where the anomaly should occur
#     mask = (df_copy.index >= pd.to_datetime(start_time)) & (df_copy.index <= pd.to_datetime(end_time))

#     # Debug: Print the number of affected rows
#     num_affected = mask.sum()
#     logger.info(f"Number of rows affected by the anomaly: {num_affected}")

#     if num_affected == 0:
#         logger.warning("No rows found in the specified time range. Check start_time and end_time.")
#         return df_copy

#     # Debug: Print the first few rows of the DataFrame to confirm the mask
#     logger.debug(f"Masked DataFrame rows:\n{df_copy.loc[mask].head()}")

#     # Apply an exponential increase to the specified variable within the specified time range
#     df_copy.loc[mask, variable] += np.exp(np.linspace(0, increase_rate * num_affected, num_affected)) - 1

#     # Debug: Check if the anomaly was applied
#     logger.debug(f"Modified DataFrame with anomalies:\n{df_copy.loc[mask].head()}")

#     return df_copy

from datetime import datetime
import pandas as pd
import numpy as np

def generate_stable_toy_data(start_date: datetime, end_date: datetime, seed_for_random: int = None) -> pd.DataFrame:
    """
    Generate a stable toy dataset for an industrial pump system with correlated variables.

    Parameters:
    -----------
    start_date : datetime
        The start datetime of the time series.
    end_date : datetime
        The end datetime of the time series.
    seed_for_random : int, optional
        Seed for random number generation for reproducibility. Defaults to None.

    Returns:
    --------
    pd.DataFrame
        A DataFrame containing the generated time series data with a 'Timestamp' index.
    """
    # Set the seed for reproducibility
    if seed_for_random is not None:
        np.random.seed(seed_for_random)
    
    # Generate a date range
    date_range = pd.date_range(start=start_date, end=end_date, freq='30s', tz='UTC')
    number_of_rows = len(date_range)
    
    # Generate base data with correlations
    temperature = np.random.normal(loc=75, scale=1, size=number_of_rows)
    pressure = 3 - 0.01 * (temperature - 75) + np.random.normal(loc=0, scale=0.05, size=number_of_rows)
    flow_rate = 300 + 10 * (3 - pressure) + np.random.normal(loc=0, scale=5, size=number_of_rows)
    vibration = 0.1 * np.sqrt(flow_rate * pressure) + np.random.normal(loc=0, scale=0.05, size=number_of_rows)
    humidity = np.random.normal(loc=40, scale=5, size=number_of_rows)
    
    # Create a DataFrame
    df = pd.DataFrame({
        'Timestamp': date_range,
        'Temperature_C': temperature,
        'Pressure_MPa': pressure,
        'Vibration_mm_s': vibration,
        'Flow_Rate_l_min': flow_rate,
        'Humidity_%': humidity
    })
    
    # Set Timestamp as the index
    df.set_index('Timestamp', inplace=True)
    
    return df

from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import logging

def introduce_exponential_anomalies(
    start_date: datetime,
    end_date: datetime,
    variable: str,
    increase_rate: float = 0.01,
    seed_for_random: int = None
) -> pd.DataFrame:
    """
    Generates a stable toy dataset and introduces an exponential anomaly by gradually increasing the specified variable
    over a given time period.

    Parameters:
    -----------
    variable (str): The column name of the variable to introduce the anomaly.
    start_time (datetime): Start time of the anomaly.
    end_time (datetime): End time of the anomaly.
    increase_rate (float): The rate at which the variable increases exponentially over time. Default is 0.01.
    seed_for_random : int, optional
        Seed for random number generation for reproducibility. Defaults to None.

    Returns:
    --------
    pd.DataFrame: A new DataFrame with the simulated anomaly.
    """
    # Generate stable toy data
    df_stable = generate_stable_toy_data(start_date, end_date, seed_for_random)

    # Make a copy of the original DataFrame to avoid modifying it directly
    df_copy = df_stable.copy()

    # Check if the index is already in ISO datetime format
    if not pd.api.types.is_datetime64_any_dtype(df_copy.index):
        logger.warning("Index is not in datetime format. Converting to datetime.")
        df_copy.index = pd.to_datetime(df_copy.index)

    # Generate a mask for the time period where the anomaly should occur
    mask = (df_copy.index >= pd.to_datetime(start_date)) & (df_copy.index <= pd.to_datetime(end_date))

    # Debug: Print the number of affected rows
    num_affected = mask.sum()
    logger.info(f"Number of rows affected by the anomaly: {num_affected}")

    if num_affected == 0:
        logger.warning("No rows found in the specified time range. Check start_time and end_time.")
        return df_copy

    # Debug: Print the first few rows of the DataFrame to confirm the mask
    logger.debug(f"Masked DataFrame rows:\n{df_copy.loc[mask].head()}")

    # Apply an exponential increase to the specified variable within the specified time range
    df_copy.loc[mask, variable] += np.exp(np.linspace(0, increase_rate * num_affected, num_affected)) - 1

    # Debug: Check if the anomaly was applied
    logger.debug(f"Modified DataFrame with anomalies:\n{df_copy.loc[mask].head()}")

    return df_copy



def simulate_broken_sensor(df: pd.DataFrame, variable: str, start_time: str, end_time: str, mode: str = 'stuck', value: float = None) -> pd.DataFrame:
    """
    Simulates a broken sensor by introducing anomalies into the specified variable.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing the stable time series data.
    variable (str): The column name of the variable to simulate the sensor failure on.
    start_time (str): The start time of the sensor failure in 'YYYY-MM-DD HH:MM:SS' format.
    end_time (str): The end time of the sensor failure in 'YYYY-MM-DD HH:MM:SS' format.
    mode (str): The type of sensor failure. Options include 'stuck', 'jump', 'spike', and 'dropout'.
    value (float): The value to use for 'stuck' or 'jump' modes. Defaults to None, in which case the function will determine a value.
    
    Returns:
    pd.DataFrame: A new DataFrame with the simulated sensor failure.
    """
    # Make a copy of the original DataFrame to avoid modifying it directly
    df_copy = df.copy()

    # Check if the index is already in ISO datetime format
    if not pd.api.types.is_datetime64_any_dtype(df_copy.index):
        logger.warning("Index is not in datetime format. Converting to datetime.")
        df_copy.index = pd.to_datetime(df_copy.index)

    # Generate a mask for the time period where the anomaly should occur
    mask = (df_copy.index >= pd.to_datetime(start_time)) & (df_copy.index <= pd.to_datetime(end_time))

    # Simulate different types of sensor failures
    if mode == 'stuck':
        # Sensor gets stuck at a constant value
        if value is None:
            value = df_copy[variable][mask].mean()  # Use the mean value during the period as the stuck value
        df_copy.loc[mask, variable] = value

    elif mode == 'jump':
        # Sensor jumps to an unusually high or low value
        if value is None:
            value = df_copy[variable].mean() + 15 * df_copy[variable].std()  # Jump 5 standard deviations above the mean
        df_copy.loc[mask, variable] = value

    elif mode == 'spike':
        # Sensor produces intermittent spikes
        spike_indices = df_copy.loc[mask].sample(frac=0.1).index  # Select 10% of the time points within the period
        df_copy.loc[spike_indices, variable] = df_copy[variable].mean() + 15 * df_copy[variable].std()

    elif mode == 'dropout':
        # Sensor stops reporting data (NaN values)
        df_copy.loc[mask, variable] = np.nan

    return df_copy
