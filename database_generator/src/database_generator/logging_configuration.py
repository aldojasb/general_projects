import json
import os
import logging
from database_generator.helpers import get_config_path

def setup_logging(log_file_path=None, log_level=logging.INFO):
    """
    Setup logging configuration.
    
    Args:
        log_file_path (str): Path to the log file. If None, logs will be displayed on the console.
        log_level (int): Logging level. Default is logging.INFO.
    """
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    
    if log_file_path:
        # Ensure the log directory exists
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        
        logging.basicConfig(
            level=log_level,
            format=log_format,
            datefmt=date_format,
            handlers=[
                logging.FileHandler(log_file_path),
                logging.StreamHandler()
            ]
        )
    else:
        logging.basicConfig(
            level=log_level,
            format=log_format,
            datefmt=date_format
        )


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

    ##### Step 2: Construct the dynamic log file path
    log_file_path = os.path.join(
        path_to_save_the_outcomes,
        "tmp",
        "log_records",
        variable_of_interest,
        "anomaly_detection_app.log"
    )

    # Ensure the log directory exists
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    # Step 3: Setup logging with the dynamic path
    setup_logging(log_file_path=log_file_path)
    
    logging.getLogger(__name__).info("Logging is set up correctly.")

