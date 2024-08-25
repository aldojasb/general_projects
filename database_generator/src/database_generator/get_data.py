import numpy as np
import pandas as pd
import logging

def generate_stable_toy_data(number_of_rows: int, start_date: str, seed_for_random: int = None) -> pd.DataFrame:
    # Set the seed for reproducibility
    if seed_for_random is not None:
        np.random.seed(seed_for_random)
    
    # Generate a date range
    date_range = pd.date_range(start=start_date, periods=number_of_rows, freq='5min', tz='UTC')
    
    # Generate base data with correlations
    # Temperature: Normally distributed around 75°C with small fluctuations
    temperature = np.random.normal(loc=75, scale=1, size=number_of_rows)
    
    # Pressure: Correlated with temperature, slightly decreasing with higher temperatures
    pressure = 3 - 0.01 * (temperature - 75) + np.random.normal(loc=0, scale=0.05, size=number_of_rows)
    
    # Flow Rate: Generally stable, slightly increasing with lower pressure (inverse correlation)
    flow_rate = 300 + 10 * (3 - pressure) + np.random.normal(loc=0, scale=5, size=number_of_rows)
    
    # Vibration: Non-linear increase with flow_rate and pressure
    vibration = 0.1 * np.sqrt(flow_rate * pressure) + np.random.normal(loc=0, scale=0.05, size=number_of_rows)
    
    # Humidity: Independent of the other variables, normal fluctuations
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


def introduce_exponential_anomalies(df: pd.DataFrame, variable: str, start_time: str, end_time: str, increase_rate: float = 0.01) -> pd.DataFrame:
    """
    Introduces an exponential anomaly by gradually increasing the specified variable over a given time period.

    Parameters:
    df (pd.DataFrame): DataFrame containing the stable time series data.
    variable (str): The column name of the variable to introduce the anomaly.
    start_time (str): The start time of the anomaly in 'YYYY-MM-DD HH:MM:SS' format.
    end_time (str): The end time of the anomaly in 'YYYY-MM-DD HH:MM:SS' format.
    increase_rate (float): The rate at which the variable increases exponentially over time. Default is 0.01.

    Returns:
    pd.DataFrame: A new DataFrame with the simulated anomaly.
    """
    # Make a copy of the original DataFrame to avoid modifying it directly
    df_copy = df.copy()

    # Check if the index is already in ISO datetime format
    if not pd.api.types.is_datetime64_any_dtype(df_copy.index):
        logging.warning("Index is not in datetime format. Converting to datetime.")
        df_copy.index = pd.to_datetime(df_copy.index)

    # Generate a mask for the time period where the anomaly should occur
    mask = (df_copy.index >= pd.to_datetime(start_time)) & (df_copy.index <= pd.to_datetime(end_time))

    # Get the number of affected rows
    num_affected = mask.sum()

    # Apply an exponential increase to the specified variable within the specified time range
    df_copy.loc[mask, variable] += np.exp(np.linspace(0, increase_rate * num_affected, num_affected)) - 1

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
        logging.warning("Index is not in datetime format. Converting to datetime.")
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
