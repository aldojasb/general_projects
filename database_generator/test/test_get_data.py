import pytest
from database_generator.get_data import generate_stable_toy_data
from datetime import timedelta
import pandas as pd

def test_generate_stable_toy_data_returns_three_rows_for_one_minute_range():
    """
    Test that the generate_stable_toy_data function returns three rows for a 
    time range of 1 minute with a frequency of 30 seconds.
    """
    # Givven:
    start_time = pd.Timestamp(year=2024,month=10, day=5, hour=16, minute=0, second=0, microsecond=0,tz='utc')
    end_time = start_time + timedelta(minutes=1)
    frequency = '30s'
    random_state = 42
    
    # When
    df = generate_stable_toy_data(
        start_datetime=start_time,
        end_datetime=end_time,
        frequency=frequency,
        seed_for_random=random_state,
    )
    
    # Then
    assert df.shape[0] == 3

def test_generate_stable_toy_data_raises_error_for_wrong_datetime_order():
    """
    Test if the generate_stable_toy_data function raises a ValueError when the 
    start datetime is after the end datetime.
    """
    # Given
    start_time = pd.Timestamp(year=2024, month=10, day=5, hour=16, second=0, microsecond=0, tz='utc')
    end_time = start_time - timedelta(minutes=1)
    frequency = '30s'
    random_state = 42

    # When / Then
    with pytest.raises(ValueError, match='must be after start_datetime'):
            df = generate_stable_toy_data(
                 start_datetime=start_time,
                 end_datetime=end_time,
                 frequency=frequency,
                 seed_for_random=random_state
                 )

def test_validate_time_range_raises_error_for_short_time_range():
    """
    Test that validate_time_range raises a ValueError when the time range is shorter
    than the specified frequency.
    """
    # Given
    start_time = pd.Timestamp(year=2024, month=10, day=5, hour=16, minute=0, second=0, tz='UTC')
    end_time = start_time + timedelta(seconds=10)  # Time range is shorter than 30 seconds
    frequency = '30s'
    random_state = 42
    # When / Then
    with pytest.raises(ValueError, match="Time range is too short"):
            df = generate_stable_toy_data(
                 start_datetime=start_time,
                 end_datetime=end_time,
                 frequency=frequency,
                 seed_for_random=random_state
                 )

