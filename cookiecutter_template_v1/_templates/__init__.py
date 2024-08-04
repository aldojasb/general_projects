# __init__.py

import json
import os
import logging
from datetime import datetime
from .logging_configuration import setup_logging
from .config_path import get_config_path
import argparse


# Step 0: Get the path to the configuration file
path_for_the_json_file = get_config_path()

# Step 1: Read variables from the JSON file
with open(path_for_the_json_file, 'r') as file:
    params = json.load(file)


training_parameters = params["parameters_to_train_the_model"]
variable_of_interest = training_parameters["variable_of_interest"]
path_to_save_the_outcomes = training_parameters["path_to_save_the_outcomes"]


# Step 2: Generate the timestamp
global timestamp_for_this_experiment
timestamp_for_this_experiment = datetime.now().strftime("%Y%m%d_%H%M%S")

# Step 3: Construct the dynamic log file path
log_file_path = os.path.join(
    path_to_save_the_outcomes,
    "tmp",
    "outcomes_from_the_model",
    variable_of_interest,
    timestamp_for_this_experiment,
    "logs",
    "anomaly_detection_app.log"
)

# Ensure the log directory exists
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

# Step 4: Setup logging with the dynamic path
setup_logging(log_file_path=log_file_path)

# Import timestamp_for_this_experiment for package-wide
__all__ = ['timestamp_for_this_experiment']