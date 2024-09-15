# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: database_generator
#     language: python
#     name: database_generator
# ---

# +
import numpy as np
import pandas as pd
from typing import Optional
from datetime import datetime, timezone, timedelta
import sys
import argparse
import os
import warnings
from typing import Optional, Literal, NewType
import json

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

from database_generator.evaluate import (
    overlaid_plots_with_plotly,
)

from database_generator.evaluate import (
    overlaid_plots_with_plotly,
)

from database_generator.db_operations import (
    create_sql_alchemy_engine,
    get_last_timestamp,
    query_data_by_datetime,
    store_pandas_dataframe_into_postegre,
)

# -

# # Accessing and reading the config file

# +
# get the path to the .json file from the environment

path_for_the_json_file = get_config_path()
path_for_the_json_file


# +
config_dict = load_and_process_params(path_for_the_json_file)

seed_for_the_stable_dataset = config_dict["seed_for_the_stable_dataset"]

# -

# # Create the stable data

# +
# Example usage

main_datetime_in_utc = pd.Timestamp.now(tz="UTC")
start_datetime_in_utc = main_datetime_in_utc - timedelta(hours=24)
# -

# list of dataframes
dfs_list = list()

df_stable = generate_stable_toy_data(
    start_datetime=start_datetime_in_utc,
    end_datetime=main_datetime_in_utc,
    seed_for_random=seed_for_the_stable_dataset,
)


# adding df:
dfs_list.append(df_stable)

# ### visualize the generated data

fig_stable = overlaid_plots_with_plotly(
    df=df_stable,
    # scatter_variables=['Vibration_mm_s', 'Flow_Rate_l_min'],
    # variable_of_interest='Temperature_C',
    save_plot=False,
)

fig_stable.show()

# # Create the two types of anomaly to evalaute it
#
# ### Problem 1: Bearing Wear
# Description: Over time, the bearings in the pump might wear out, causing an increase in vibration levels.
#
#
# ### Problem 5: Broken Temperature Sensor
# Description: The temperature sensor might malfunction or break, leading to inaccurate or stuck readings.
#
# - Stuck Readings: The sensor gets "stuck" at a constant value, providing the same reading for a period of time.
#
# - Sudden Jumps: The sensor might suddenly jump to an unusually high or low value, remaining there for some time.
#
# - Intermittent Spikes: The sensor occasionally produces spikes of incorrect readings, either very high or very low.
#
# - Dropouts: The sensor might stop reporting data altogether, which could be simulated as missing values (NaN).

# ### Bearing Wear Anomaly

# defining datetime
start_time_anomaly_exponential = main_datetime_in_utc
end_time_anomaly_exponential = main_datetime_in_utc + timedelta(hours=4)

# +
# Introduce bearing wear

df_with_anomaly_exponential = introduce_exponential_anomalies(
    variable="Vibration_mm_s",
    start_datetime=start_time_anomaly_exponential,
    end_datetime=end_time_anomaly_exponential,
    increase_rate=0.01,
)

dfs_list.append(df_with_anomaly_exponential)

# +
# Plot the data to see the effect of the anomaly
fig_anomaly_exponential = overlaid_plots_with_plotly(
    df_with_anomaly_exponential,
    # scatter_variables=['Vibration_mm_s'],
    variable_of_interest="Vibration_mm_s",
    save_plot=False,
)


# -

fig_anomaly_exponential.show()

# ### Stuck Readings

# +
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
# -

# Plot the data to see the effect of the anomaly
fig_anomaly_stuck = overlaid_plots_with_plotly(
    df_with_sensor_issue_stuck, variable_of_interest="Temperature_C", save_plot=False
)
fig_anomaly_stuck.show()


# ### Sudden Jumps

# +
start_datetime_jump_sensor = end_datetime_stuck_sensor
end_datetime_jump_sensor = start_datetime_jump_sensor + timedelta(hours=3)

# Simulate a sensor stuck at a constant value
df_with_sensor_issue_jump = simulate_broken_sensor(
    variable="Temperature_C",
    start_datetime=start_datetime_jump_sensor,
    end_datetime=end_datetime_jump_sensor,
    mode="jump",
)

dfs_list.append(df_with_sensor_issue_jump)
# -

# Plot the data to see the effect of the anomaly
fig_anomaly_stuck = overlaid_plots_with_plotly(
    df_with_sensor_issue_jump, variable_of_interest="Temperature_C", save_plot=False
)
fig_anomaly_stuck.show()


# ### Intermittent Spikes

# +
start_datetime_spike_sensor = end_datetime_jump_sensor
end_datetime_spike_sensor = start_datetime_spike_sensor + timedelta(hours=3)

# Simulate a sensor stuck at a constant value
df_with_sensor_issue_spike = simulate_broken_sensor(
    variable="Temperature_C",
    start_datetime=start_datetime_spike_sensor,
    end_datetime=end_datetime_spike_sensor,
    mode="spike",
)

dfs_list.append(df_with_sensor_issue_spike)
# -

# Plot the data to see the effect of the anomaly
fig_anomaly_stuck = overlaid_plots_with_plotly(
    df_with_sensor_issue_spike, variable_of_interest="Temperature_C", save_plot=False
)
fig_anomaly_stuck.show()


# +
start_datetime_dropout_sensor = end_datetime_spike_sensor
end_datetime_dropout_sensor = start_datetime_dropout_sensor + timedelta(hours=3)

# Simulate a sensor stuck at a constant value
df_with_sensor_issue_dropout = simulate_broken_sensor(
    variable="Temperature_C",
    start_datetime=start_datetime_dropout_sensor,
    end_datetime=end_datetime_dropout_sensor,
    mode="dropout",
)

dfs_list.append(df_with_sensor_issue_dropout)
# -

# Plot the data to see the effect of the anomaly
fig_anomaly_stuck = overlaid_plots_with_plotly(
    df_with_sensor_issue_dropout, variable_of_interest="Temperature_C", save_plot=False
)
fig_anomaly_stuck.show()


# Concatenate DataFrames and handle duplicates by taking the mean of values for duplicate indices
combined_df = append_and_concatenate_dataframes(dfs_list, method="first")

# Store pandas df into postgre
store_pandas_dataframe_into_postegre(df=combined_df)
