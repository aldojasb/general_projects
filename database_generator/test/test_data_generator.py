import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from database_generator.data_generator import IndustrialPumpData, ExponentialAnomaly, IntermittentSpikeAnomaly

class TestIndustrialPumpData:
    """ Test suite for IndustrialPumpData """

    def test_generate_standard_data(self):
        # Given: An instance of IndustrialPumpData with valid inputs
        start_datetime = datetime(2023, 1, 1, 0, 0, tzinfo=timezone.utc)
        end_datetime = datetime(2023, 1, 2, 0, 0, tzinfo=timezone.utc)
        frequency = '1h'
        seed_for_random = 42
        pump_data = IndustrialPumpData(
            start_datetime=start_datetime,
            end_datetime=end_datetime,
            frequency=frequency,
            seed_for_random=seed_for_random
        )

        # When: The generate_standard_data method is called
        result_df = pump_data.generate_standard_data()

        # Then: The resulting DataFrame should have the correct structure and content
        assert isinstance(result_df, pd.DataFrame), "Result should be a DataFrame"
        assert len(result_df) == 25, "There should be 25 rows for an hourly frequency over 1 day"
        assert all(
            col in result_df.columns for col in ['temperature_c', 'pressure_mpa', 'vibration_mm_s',
                                                 'flow_rate_l_min','humidity_%', 'flag_normal_data']
        ), "All expected columns should be present"

class TestExponentialAnomaly:
    """ Test suite for ExponentialAnomaly """

    def test_introduce_anomaly(self):
        # Given: A standard dataset and an ExponentialAnomaly instance
        data = pd.DataFrame({
            "timestamp": pd.date_range("2023-01-01", "2023-01-02", freq="h", tz='utc'),
            "temperature": np.random.normal(75, 1, 25),
            "flag_normal_data": True
        }).set_index("timestamp")

        anomaly = ExponentialAnomaly(
            start_datetime=datetime(2023, 1, 1, 5, 0, tzinfo=timezone.utc),
            end_datetime=datetime(2023, 1, 1, 10, 0, tzinfo=timezone.utc),
            variable_to_insert_anomalies="temperature",
            standard_data=data,
            increase_rate=0.05,
            seed_for_random=42
        )

        # When: The introduce_anomaly method is called
        result_df = anomaly.introduce_anomaly()

        # Then: The returned DataFrame should contain only affected rows
        assert isinstance(result_df, pd.DataFrame), "Result should be a DataFrame"
        assert not result_df["flag_normal_data"].all(), "All rows in the result should have flag_normal_data set to False"
        assert len(result_df) == 6, "The result should contain 6 affected rows"

class TestIntermittentSpikeAnomaly:
    """ Test suite for IntermittentSpikeAnomaly """

    def test_introduce_anomaly(self):
        # Given: A standard dataset and an IntermittentSpikeAnomaly instance
        data = pd.DataFrame({
            "timestamp": pd.date_range("2023-01-01", "2023-01-02", freq="h", tz='utc'),
            "temperature": np.random.normal(75, 1, 25),
            "flag_normal_data": True
        }).set_index("timestamp")

        anomaly = IntermittentSpikeAnomaly(
            start_datetime=datetime(2023, 1, 1, 0, 0, tzinfo=timezone.utc),
            end_datetime=datetime(2023, 1, 1, 20, 0, tzinfo=timezone.utc),
            variable_to_insert_anomalies="temperature",
            standard_data=data,
            spike_fraction=0.1,
            spike_multiplier=15.0,
            seed_for_random=42
        )

        # When: The introduce_anomaly method is called
        result_df = anomaly.introduce_anomaly()

        # Then: The returned DataFrame should contain only affected rows
        assert isinstance(result_df, pd.DataFrame), "Result should be a DataFrame"
        assert not result_df["flag_normal_data"].all(), "All rows in the result should have flag_normal_data set to False"
        assert len(result_df) == 2, "The result should contain 2 affected rows for 10% of 20 rows in range"
