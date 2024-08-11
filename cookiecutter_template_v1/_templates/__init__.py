# __init__.py

import json
import os
import logging
from datetime import datetime
from .logging_configuration import setup_logging
from .config_path import get_config_path
import argparse

# # Function to log debug messages
# def debug_log(message):
#     print(message)
#     logging.getLogger(__name__).debug(message)

##### Step 0: Get the path to the configuration file

# debug_log("Getting the path to the configuration file.")
path_for_the_json_file = get_config_path()
# debug_log(f"Configuration file path: {path_for_the_json_file}")

##### Step 1: Read variables from the JSON file

# debug_log("Reading variables from the JSON file.")
with open(path_for_the_json_file, 'r') as file:
    params = json.load(file)
# debug_log(f"Parameters read from JSON: {params}")

training_parameters = params["parameters_to_train_the_model"]
variable_of_interest = training_parameters["variable_of_interest"]

parameters_to_save_the_outcomes = params["parameters_to_save_the_outcomes"]
path_to_save_the_outcomes = parameters_to_save_the_outcomes["path_to_save_the_outcomes"]
# debug_log(f"Path to save the outcomes: {path_to_save_the_outcomes}")

##### Step 2: Generate the timestamp

# debug_log("Generating the timestamp for this experiment.")
global timestamp_for_this_experiment
timestamp_for_this_experiment = datetime.now().strftime("%Y%m%d_%H%M%S")
# debug_log(f"Timestamp for this experiment: {timestamp_for_this_experiment}")

##### Step 3: Construct the dynamic log file path
# debug_log("Constructing the dynamic log file path.")
log_file_path = os.path.join(
    path_to_save_the_outcomes,
    "tmp",
    "outcomes_from_the_model",
    variable_of_interest,
    timestamp_for_this_experiment,
    "logs",
    "anomaly_detection_app.log"
)
# debug_log(f"Log file path: {log_file_path}")

# Ensure the log directory exists
log_dir = os.path.dirname(log_file_path)
# debug_log(f"Ensuring log directory exists: {log_dir}")
# try:
#     os.makedirs(log_dir, exist_ok=True)
#     debug_log(f"Log directory created successfully or already exists: {log_dir}")
# except Exception as e:
#     debug_log(f"Failed to create log directory: {e}")

##### Step 4: Setup logging with the dynamic path
# debug_log("Setting up logging with the dynamic path.")
setup_logging(log_file_path=log_file_path)

# Test log message to verify setup
logging.getLogger(__name__).info("Logging is set up correctly.")

# Import timestamp_for_this_experiment for package-wide
__all__ = ['timestamp_for_this_experiment']
