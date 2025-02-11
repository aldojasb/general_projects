# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python (Poetry DBToolkit)
#     language: python
#     name: poetry-dbtoolkit
# ---

# # DEMO version

# ### Set up the environmental variables
# It's recommended to create a ".env" file to setup the logger configuration
#
# You must set the 'PATH_TO_SAVE_THE_OUTCOMES' as an env variable
# Example:
# ###### PATH_TO_SAVE_THE_OUTCOMES=/workspace/general_projects/database_toolkit/notebooks/tmp
#

# +
from dotenv import load_dotenv
import os

# Load the .env file only if it exists
dotenv_path = '/workspace/general_projects/database_generator/.env'
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)
    print(f"Loaded environment variables from {dotenv_path}")
else:
    print(f"No .env file found at {dotenv_path}, relying on system environment variables.")

# # Access the environment variable, with a fallback
# path_to_logs = os.getenv('PATH_TO_SAVE_THE_LOGS')
# print(f"Logs will be saved to: {path_to_logs}")

# Access the environment variable, with a fallback
path_to_logs = os.getenv('PATH_TO_SAVE_THE_OUTCOMES')
print(f"Logs will be saved to: {path_to_logs}")
# -

print(os.environ)

# 1. Generating Standard Data
# Create a dataset representing industrial pump operations.

# +
from database.toolkit.data_generator import IndustrialPumpData
from datetime import datetime, timezone

start_datetime = datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc)
end_datetime = datetime(2025, 1, 4, 0, 0, tzinfo=timezone.utc)
frequency = '1h'
seed_for_random = 42
data_generator = IndustrialPumpData(
    start_datetime=start_datetime,
    end_datetime=end_datetime,
    frequency=frequency,
    seed_for_random=seed_for_random
)

# -

standard_pump_data = data_generator.generate_standard_data()

standard_pump_data.columns

# 2. Introducing Anomalies
# Apply an exponential anomaly to simulate increasing deviations.

# +
from database.toolkit.data_generator import ExponentialAnomaly

exponential_anomaly_in_pressure = ExponentialAnomaly(
    start_datetime= datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc),
    end_datetime= datetime(2025, 1, 1, 23, 0, tzinfo=timezone.utc),
    variable_to_insert_anomalies='pressure_mpa',
    standard_data=standard_pump_data
)
# -

anomalous_data = exponential_anomaly_in_pressure.introduce_anomaly()

anomalous_data.tail()

anomalous_data.head()

# Apply an intermittent spike anomaly to simulate sudden outliers.

# +
# Create an IntermittentSpikeAnomaly instance
from database.toolkit.data_generator import IntermittentSpikeAnomaly

spike_anomaly = IntermittentSpikeAnomaly(
    start_datetime=datetime(2025, 1, 2, 0, 0, tzinfo=timezone.utc),
    end_datetime=datetime(2025, 1, 2, 23, 59, tzinfo=timezone.utc),
    variable_to_insert_anomalies="pressure_mpa",
    standard_data=standard_pump_data,
    spike_fraction=0.30,
    spike_multiplier=100.0,
    seed_for_random=42
)

# Introduce anomalies
spike_data = spike_anomaly.introduce_anomaly()
# -

spike_data

# 3. Creating a Combined Database
# Merge multiple datasets while prioritizing anomalous records.

# +
from database.toolkit.data_generator import SimpleDatabaseFactory
list_of_dfs = [standard_pump_data,
               anomalous_data,
               spike_data]

factory = SimpleDatabaseFactory(list_of_df=list_of_dfs, flag_column='flag_normal_data')
final_database = factory.create_database()
# -

final_database.head()

final_database.tail()


