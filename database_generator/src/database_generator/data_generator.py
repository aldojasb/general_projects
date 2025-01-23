from abc import ABC, abstractmethod
from datetime import datetime
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
import logging

# Set up logging
from database_generator.logging_configuration import setup_logging_for_this_script
setup_logging_for_this_script()
logger = logging.getLogger(__name__)

class StandardDataGenerator(ABC):
    """
    Abstract base class for generating standard datasets.
    Provides an interface for creating data with defined frequency and randomization.
    """
    @abstractmethod
    def generate_standard_data(self) -> pd.DataFrame:
        """
        Generate a standard dataset.

        Returns:
        - pd.DataFrame: A DataFrame containing the generated data.
        """
        pass


class AnomalyGenerator(ABC):
    """
    Abstract base class for generating anomalies in datasets.
    Provides a unified interface for anomaly introduction.
    """
    @abstractmethod
    def introduce_anomaly(self) -> pd.DataFrame:
        """
        Introduce anomalies in the dataset.

        Returns:
        - pd.DataFrame: A DataFrame containing the data with introduced anomalies.
        """
        pass


class DatabaseFactory(ABC):
    """
    Abstract base class for creating a database from a collection of dataframes.
    Provides an interface to combine or process multiple dataframes into a single dataset.
    """
    list_of_df: list[pd.DataFrame]

    @abstractmethod
    def create_database(self) -> pd.DataFrame:
        """
        Create a database by combining or processing the provided dataframes.

        Returns:
        - pd.DataFrame: A DataFrame representing the created database.
        """
        pass

@dataclass
class IndustrialPumpData(StandardDataGenerator):
    """
    Concrete implementation of StandardDataGenerator for industrial pump data.

    Attributes:
    - start_datetime (datetime): Start time for data generation.
    - end_datetime (datetime): End time for data generation.
    - frequency (str): Frequency of data points (e.g., '1H', '30T').
    - seed_for_random (int, optional): Seed for random number generator to ensure reproducibility.
    - include_flag (bool, optional): Whether to include a flag column indicating normal data (default: True).
    """
    start_datetime: datetime
    end_datetime: datetime
    frequency: str
    seed_for_random: int = field(default=None)
    include_flag: bool = field(default=True)

    def generate_standard_data(self) -> pd.DataFrame:
        """
        Generate industrial pump data over a specified time range with defined frequency.

        Returns:
        - pd.DataFrame: A DataFrame containing generated pump data with columns for temperature, pressure,
          vibration, flow rate, and humidity.
        """
        # # Validate and convert datetime inputs
        # start_datetime = validate_and_convert_datetime(self.start_datetime)
        # end_datetime = validate_and_convert_datetime(self.end_datetime)

        # # Validate that end_datetime is after start_datetime
        # validate_datetime_order(start_datetime, end_datetime)

        # # Check that the time range is larger than or equal to the frequency
        # validate_time_range(start_datetime, end_datetime, self.frequency)

        # Set the seed for reproducibility
        if self.seed_for_random is not None:
            np.random.seed(self.seed_for_random)

        # Generate a date range
        date_range = pd.date_range(
            start=self.start_datetime, end=self.end_datetime, freq=self.frequency, tz="UTC"
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
                "timestamp": date_range,
                "temperature_c": temperature,
                "pressure_mpa": pressure,
                "vibration_mm_s": vibration,
                "flow_rate_l_min": flow_rate,
                "humidity_%": humidity,
            }
        )

        if self.include_flag:
            df["flag_normal_data"] = True  # Add the flag column conditionally

        # Set Timestamp as the index
        df.set_index("timestamp", inplace=True)

        # logger info
        logger.info(f"A new standard dataset was created")

        return df


@dataclass
class ExponentialAnomaly(AnomalyGenerator):
    """
    Introduces an exponential anomaly to a specified variable in a dataset
    within a given time range. Updates the `flag_normal_data` to False
    for affected rows.

    Attributes:
    - start_datetime (datetime): Start time for the anomaly.
    - end_datetime (datetime): End time for the anomaly.
    - variable_to_insert_anomalies (str): The column to apply the anomaly to.
    - standard_data (pd.DataFrame): The original dataset to modify.
    - increase_rate (float, optional): The rate of exponential increase (default: 0.01).
    - seed_for_random (int, optional): Seed for random number generator to ensure reproducibility.
    """
    start_datetime: datetime
    end_datetime: datetime
    variable_to_insert_anomalies: str
    standard_data: pd.DataFrame
    increase_rate: float = field(default=0.01)
    seed_for_random: int = field(default=None)

    def introduce_anomaly(self) -> pd.DataFrame:
        # Validate the column exists
        if self.variable_to_insert_anomalies not in self.standard_data.columns:
            raise ValueError(f"Column '{self.variable_to_insert_anomalies}' not found in the dataset.")

        # Make a copy of the original DataFrame to avoid modifying it directly
        df_copy = self.standard_data.copy()

        # Generate a mask for the time period where the anomaly should occur
        mask = (df_copy.index >= self.start_datetime) & (df_copy.index <= self.end_datetime)

        # Debug: Print the number of affected rows
        num_affected = mask.sum()
        logger.debug(f"Number of rows affected by the anomaly: {num_affected}")

        if num_affected == 0:
            logger.warning(
                "No rows found in the specified time range. Check start_datetime and end_datetime."
            )
            return self.standard_data

        # Set the random seed for reproducibility
        if self.seed_for_random is not None:
            np.random.seed(self.seed_for_random)

        # Filter only the rows that will be affected by the anomaly
        df_filtered = df_copy.loc[mask]

        # Apply an exponential increase to the specified variable
        affected_values = df_filtered.loc[: , self.variable_to_insert_anomalies]
        anomalies = (
            np.exp(np.linspace(0, self.increase_rate * num_affected, num_affected)) - 1
        )
        df_filtered.loc[: , self.variable_to_insert_anomalies] = affected_values + anomalies

        # Update the flag column
        df_filtered.loc[: , "flag_normal_data"] = False

        # logger info
        logger.info(f" A new dataset with {num_affected} anomalies was created")

        # Return only the modified columns and affected rows
        return df_filtered


@dataclass
class IntermittentSpikeAnomaly(AnomalyGenerator):
    """
    Introduces intermittent spike anomalies to a specified variable in a dataset
    within a given time range. Updates the `flag_normal_data` to False
    for affected rows.

    Attributes:
    - start_datetime (datetime): Start time for the anomaly.
    - end_datetime (datetime): End time for the anomaly.
    - variable_to_insert_anomalies (str): The column to apply the anomaly to.
    - standard_data (pd.DataFrame): The original dataset to modify.
    - spike_fraction (float, optional): Fraction of rows within the time range to apply spikes (default: 0.1).
    - spike_multiplier (float, optional): Multiplier for the spike value based on the variable's standard deviation (default: 15).
    - seed_for_random (int, optional): Seed for random number generator to ensure reproducibility.
    """
    start_datetime: datetime
    end_datetime: datetime
    variable_to_insert_anomalies: str
    standard_data: pd.DataFrame
    spike_fraction: float = field(default=0.1)
    spike_multiplier: float = field(default=15.0)
    seed_for_random: int = field(default=None)

    def introduce_anomaly(self) -> pd.DataFrame:
        # Validate the column exists
        if self.variable_to_insert_anomalies not in self.standard_data.columns:
            raise ValueError(f"Column '{self.variable_to_insert_anomalies}' not found in the dataset.")

        # Make a copy of the original DataFrame to avoid modifying it directly
        df_copy = self.standard_data.copy()

        # Generate a mask for the time period where the anomaly should occur
        mask = (df_copy.index >= self.start_datetime) & (df_copy.index <= self.end_datetime)

        # Debug: Print the number of rows in the time range
        num_affected = mask.sum()
        logger.debug(f"Number of rows in the specified time range: {num_affected}")

        if num_affected == 0:
            logger.warning(
                "No rows found in the specified time range. Check start_datetime and end_datetime."
            )
            return pd.DataFrame()

        # Set the random seed for reproducibility
        if self.seed_for_random is not None:
            np.random.seed(self.seed_for_random)

        # Filter the rows within the time range
        df_filtered = df_copy.loc[mask]

        # Calculate the exact number of spikes
        num_spikes = min(len(df_filtered), max(1, int(self.spike_fraction * num_affected)))

        # Ensure the DataFrame has enough rows to sample
        if len(df_filtered) < num_spikes:
            logger.warning("Not enough rows to generate the required number of spikes. Adjusting spike count.")
            num_spikes = len(df_filtered)

        # Select random rows for introducing spikes
        spike_indices = df_filtered.sample(n=num_spikes, random_state=self.seed_for_random).index

        # Apply spike anomalies to the selected rows
        spike_value = df_filtered[self.variable_to_insert_anomalies].mean() + \
                    self.spike_multiplier * df_filtered[self.variable_to_insert_anomalies].std()
        df_copy.loc[spike_indices, self.variable_to_insert_anomalies] = spike_value

        # Update the flag column
        df_copy.loc[spike_indices, "flag_normal_data"] = False

        # Log information about the anomaly
        logger.info(f"Introduced {num_spikes} spike anomalies in '{self.variable_to_insert_anomalies}'.")

        # Return only the affected rows (all columns)
        return df_copy.loc[spike_indices]


@dataclass
class SimpleDatabaseFactory:
    """
    Concrete implementation of DatabaseFactory for creating a database by concatenating
    multiple DataFrames. Resolves conflicts for duplicate indices by retaining the row
    where a specified flag column (default: 'flag_normal_data') is False.

    Attributes:
    - list_of_df (list[pd.DataFrame]): A list of DataFrames to combine into a single dataset.
    - flag_column (str): The name of the column used for conflict resolution (default: 'flag_normal_data').
    """
    list_of_df: list[pd.DataFrame]
    flag_column: str

    def create_database(self) -> pd.DataFrame:
        """
        Create a database by concatenating the provided DataFrames and resolving conflicts
        for duplicate indices by prioritizing rows where the flag column is False.

        Returns:
        --------
        - pd.DataFrame: A DataFrame representing the created database.
        """
        if not self.list_of_df:
            raise ValueError("The list of DataFrames is empty. Cannot create a database.")

        # Concatenate all DataFrames
        combined_df = pd.concat(self.list_of_df, axis=0)

        # Handle duplicate indices
        if combined_df.index.duplicated().any():
            # Sort rows such that 'flag_column=False' appears first for each index
            combined_df = combined_df.sort_values(by=self.flag_column, ascending=True)

            # Drop duplicates, keeping the first occurrence (where 'flag_column=False' is prioritized)
            combined_df = combined_df[~combined_df.index.duplicated(keep="first")]

        return combined_df