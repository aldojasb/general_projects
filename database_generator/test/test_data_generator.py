import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from database_generator.data_generator import IndustrialPumpData

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
