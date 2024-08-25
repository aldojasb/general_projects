# initializer.py

import json
import os
import logging
from datetime import datetime
from .logging_configuration import setup_logging
from .helpers import get_config_path


def initialize_logging_configuration():
    ##### Step 0: Get the path to the configuration file

    path_for_the_json_file = get_config_path()

    ##### Step 1: Read variables from the JSON file

    with open(path_for_the_json_file, 'r') as file:
        params = json.load(file)

    training_parameters = params["parameters_to_train_the_model"]
    variable_of_interest = training_parameters["variable_of_interest"]

    parameters_to_save_the_outcomes = params["parameters_to_save_the_outcomes"]
    path_to_save_the_outcomes = parameters_to_save_the_outcomes["path_to_save_the_outcomes"]

    ##### Step 2: Generate the timestamp

    timestamp_for_this_experiment = datetime.now().strftime("%Y%m%d_%H%M%S")

    ##### Step 3: Construct the dynamic log file path
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
    logging.getLogger(__name__).info("Logging is set up correctly.")

    # Return the timestamp for use in other parts of the package if needed
    return timestamp_for_this_experiment
