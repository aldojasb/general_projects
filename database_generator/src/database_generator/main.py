import pandas as pd
from datetime import timedelta
import logging
from database_generator.logging_configuration import setup_logging_for_this_script

setup_logging_for_this_script()
# Get the logger for this module
logger = logging.getLogger(__name__)

from database_generator.helpers import (
    get_config_path,
    load_and_process_params,
    append_and_concatenate_dataframes,
)

from database_generator.get_data import (
    generate_stable_toy_data,
    introduce_exponential_anomalies,
    simulate_broken_sensor,
)

from database_generator.db_operations import (
    store_pandas_dataframe_into_postegre,
)


# # Accessing and reading the config file
# get the path to the .json file from the environment

path_for_the_json_file = get_config_path()

config_dict = load_and_process_params(path_for_the_json_file)

seed_for_the_stable_dataset = config_dict["seed_for_the_stable_dataset"]

# list of dataframes
dfs_list = list()


### Create the stable data
main_datetime_in_utc = pd.Timestamp.now(tz="UTC")
start_datetime_in_utc = main_datetime_in_utc - timedelta(hours=24)


df_stable = generate_stable_toy_data(
    start_datetime=start_datetime_in_utc,
    end_datetime=main_datetime_in_utc,
    seed_for_random=seed_for_the_stable_dataset,
)

# adding df:
dfs_list.append(df_stable)


### Introduce bearing wear

# defining datetime
start_time_anomaly_exponential = main_datetime_in_utc
end_time_anomaly_exponential = main_datetime_in_utc + timedelta(hours=4)

df_with_anomaly_exponential = introduce_exponential_anomalies(
    variable="Vibration_mm_s",
    start_datetime=start_time_anomaly_exponential,
    end_datetime=end_time_anomaly_exponential,
    increase_rate=0.01,
)

dfs_list.append(df_with_anomaly_exponential)


# ### Stuck Readings
start_datetime_stuck_sensor = end_time_anomaly_exponential
end_datetime_stuck_sensor = start_datetime_stuck_sensor + timedelta(hours=3)

# Simulate a sensor stuck at a constant value
df_with_sensor_issue_stuck = simulate_broken_sensor(
    variable="Temperature_C",
    start_datetime=start_datetime_stuck_sensor,
    end_datetime=end_datetime_stuck_sensor,
    mode="stuck",
)

dfs_list.append(df_with_sensor_issue_stuck)


### Sudden Jumps

start_datetime_jump_sensor = end_datetime_stuck_sensor
end_datetime_jump_sensor = start_datetime_jump_sensor + timedelta(hours=3)

# Simulate a sensor jump
df_with_sensor_issue_jump = simulate_broken_sensor(
    variable="Temperature_C",
    start_datetime=start_datetime_jump_sensor,
    end_datetime=end_datetime_jump_sensor,
    mode="jump",
)

dfs_list.append(df_with_sensor_issue_jump)


### Intermittent Spikes

start_datetime_spike_sensor = end_datetime_jump_sensor
end_datetime_spike_sensor = start_datetime_spike_sensor + timedelta(hours=3)

# Simulate a sensor spike
df_with_sensor_issue_spike = simulate_broken_sensor(
    variable="Temperature_C",
    start_datetime=start_datetime_spike_sensor,
    end_datetime=end_datetime_spike_sensor,
    mode="spike",
)

dfs_list.append(df_with_sensor_issue_spike)


### Dropout sensor

start_datetime_dropout_sensor = end_datetime_spike_sensor
end_datetime_dropout_sensor = start_datetime_dropout_sensor + timedelta(hours=3)

# Simulate a sensor dropoutat a constant value
df_with_sensor_issue_dropout = simulate_broken_sensor(
    variable="Temperature_C",
    start_datetime=start_datetime_dropout_sensor,
    end_datetime=end_datetime_dropout_sensor,
    mode="dropout",
)

dfs_list.append(df_with_sensor_issue_dropout)


# Concatenate DataFrames and handle duplicates by taking the mean of values for duplicate indices
combined_df = append_and_concatenate_dataframes(dfs_list, method="first")

# Store pandas df into postgre
store_pandas_dataframe_into_postegre(df=combined_df)
