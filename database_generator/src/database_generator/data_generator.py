from abc import ABC, abstractmethod
from datetime import datetime
from dataclasses import dataclass
import pandas as pd
import numpy as np

from database_generator.helpers import (
    validate_and_convert_datetime,
    validate_datetime_order,
    validate_time_range,
)

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
    """
    start_datetime: datetime
    end_datetime: datetime
    frequency: str
    seed_for_random: int = None

    def generate_standard_data(self) -> pd.DataFrame:
        """
        Generate industrial pump data over a specified time range with defined frequency.

        Returns:
        - pd.DataFrame: A DataFrame containing generated pump data with columns for temperature, pressure,
          vibration, flow rate, and humidity.
        """
        # Validate and convert datetime inputs
        start_datetime = validate_and_convert_datetime(self.start_datetime)
        end_datetime = validate_and_convert_datetime(self.end_datetime)

        # Validate that end_datetime is after start_datetime
        validate_datetime_order(start_datetime, end_datetime)

        # Check that the time range is larger than or equal to the frequency
        validate_time_range(start_datetime, end_datetime, self.frequency)

        # Set the seed for reproducibility
        if self.seed_for_random is not None:
            np.random.seed(self.seed_for_random)

        # Generate a date range
        date_range = pd.date_range(
            start=start_datetime, end=end_datetime, freq=self.frequency, tz="UTC"
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

