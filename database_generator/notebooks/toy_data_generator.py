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
from datetime import datetime, timezone
import logging
import sys
import argparse
import os
import warnings
from typing import Optional, Literal, NewType
import json


# Get the logger for this module
logger = logging.getLogger(__name__)

# -

# import sys
# print(sys.executable)


from database_generator.helpers import (
    get_config_path,
    load_and_process_params,
)

from database_generator.get_data import (
    generate_stable_toy_data,
    introduce_exponential_anomalies,
    simulate_broken_sensor,
)

from database_generator.evaluate import (
    overlaid_plots_with_plotly,
)

from database_generator import timestamp_for_this_experiment # get global variable from __init__.py
timestamp_for_this_experiment

# # Accessing and reading the config file

# +
# get the path to the .json file from the environment

path_for_the_json_file = get_config_path()
path_for_the_json_file

# -

(
    start_date_for_the_toy_dataset,
    number_of_rows_for_stable_toy_data,
    seed_for_the_stable_dataset
    ) = load_and_process_params(path_for_the_json_file)

# # Create the stable data

# +
# Example usage
df_stable = generate_stable_toy_data(number_of_rows=number_of_rows_for_stable_toy_data, start_date=start_date_for_the_toy_dataset, seed_for_random=42)

df_stable.head()
# -

df_stable.tail()

# ### visualize the generated data

df_stable.info()

fig = overlaid_plots_with_plotly(df=df_stable,
                           scatter_variables=['Vibration_mm_s', 'Flow_Rate_l_min'],
                           variable_of_interest='Temperature_C',
                           save_plot=False)
fig.show()

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

df_stable.head()

# +
# Introduce bearing wear

df_with_anomaly = introduce_exponential_anomalies(df= df_stable,
                                                  variable='Vibration_mm_s',
                                                  start_time='2024-09-16 12:00:00+00:00',
                                                  end_time='2024-09-27 03:15:00+00:00',
                                                  increase_rate=0.0015
                                                  )

# -

# Plot the data to see the effect of the anomaly
fig = overlaid_plots_with_plotly(df_with_anomaly, scatter_variables=['Vibration_mm_s'], variable_of_interest='Temperature_C', save_plot=False)
fig.show()

# +
# Simulate a sensor stuck at a constant value
df_with_sensor_issue = simulate_broken_sensor(df= df_stable,
                                              variable='Temperature_C',
                                              start_time='2024-09-20 12:00:00+00:00',
                                              end_time='2024-09-27 03:15:00+00:00',
                                              mode='stuck'
                                              )

# Plot the data to see the effect of the anomaly
fig = overlaid_plots_with_plotly(df_with_sensor_issue, variable_of_interest='Temperature_C', save_plot=False)
fig.show()

# +
# Simulate a sensor stuck at a constant value
df_with_sensor_issue = simulate_broken_sensor(df= df_stable,
                                              variable='Temperature_C',
                                              start_time='2024-09-20 12:00:00+00:00',
                                              end_time='2024-09-27 03:15:00+00:00',
                                              mode='jump'
                                              )

# Plot the data to see the effect of the anomaly
fig = overlaid_plots_with_plotly(df_with_sensor_issue, variable_of_interest='Temperature_C', save_plot=False)
fig.show()

# +
# Simulate a sensor stuck at a constant value
df_with_sensor_issue = simulate_broken_sensor(df= df_stable,
                                              variable='Temperature_C',
                                              start_time='2024-09-20 12:00:00+00:00',
                                              end_time='2024-09-27 03:15:00+00:00',
                                              mode='spike'
                                              )

# Plot the data to see the effect of the anomaly
fig = overlaid_plots_with_plotly(df_with_sensor_issue, variable_of_interest='Temperature_C', save_plot=False)
fig.show()

# +
# Simulate a sensor stuck at a constant value
df_with_sensor_issue = simulate_broken_sensor(df= df_stable,
                                              variable='Temperature_C',
                                              start_time='2024-09-20 12:00:00+00:00',
                                              end_time='2024-09-27 03:15:00+00:00',
                                              mode='dropout'
                                              )

# Plot the data to see the effect of the anomaly
fig = overlaid_plots_with_plotly(df_with_sensor_issue, variable_of_interest='Temperature_C', save_plot=False)
fig.show()
