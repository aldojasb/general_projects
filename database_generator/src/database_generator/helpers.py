import os
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
import logging
import json

def get_config_path():
    # Check if the environment variable is set
    env_path = os.getenv('PATH_TO_THE_CONFIGURATION_FILE')
    
    if env_path:
        return env_path

    # If not, parse the command-line arguments safely
    parser = argparse.ArgumentParser(description='Provide the path to the configuration file.')
    parser.add_argument('--config', type=str, help='Path to the configuration file')
    
    # Only parse known arguments, ignore unknown ones
    args, unknown = parser.parse_known_args()
    
    if args.config:
        return args.config
    else:
        raise ValueError("Configuration file path must be provided"
                         " either as an environment variable 'PATH_TO_THE_CONFIGURATION_FILE'"
                         " or as a command-line argument '--config'.")


# Parse the timestamps with error handling
def parse_timestamp(timestamp_str: str) -> datetime:
    try:
        # Replace '-00' with '+00:00' to make it a valid ISO 8601 format
        if timestamp_str.endswith('-00'):
            timestamp_str = timestamp_str.replace('-00', '+00:00')
        return datetime.fromisoformat(timestamp_str)
    except ValueError as e:
        logging.error(f"Error parsing timestamp: {timestamp_str}. {e}"
                        "datetime should be in ISO 8601 format")
        raise


def load_and_process_params(file_path: str) -> dict:
    # Load parameters from JSON file
    with open(file_path, "r") as file:
        params = json.load(file)

    start_date_for_the_toy_dataset = parse_timestamp(params["parameters_to_create_toy_data"]["start_date_for_the_toy_dataset"])
    logging.info(f'{start_date_for_the_toy_dataset = }')

    # Access nested parameter maps under the 'parameters_to_create_toy_data' key
    seed_for_the_stable_dataset = params ["parameters_to_create_toy_data"]["seed_for_the_stable_dataset"]
    logging.info(f'{seed_for_the_stable_dataset = }')

    # Access nested parameter maps under the 'parameters_to_create_toy_data' key
    number_of_rows_for_stable_toy_data = params ["parameters_to_create_toy_data"]["number_of_rows_for_stable_toy_data"]
    logging.info(f'{number_of_rows_for_stable_toy_data = }')

    return {
        'start_date_for_the_toy_dataset': start_date_for_the_toy_dataset,
        'number_of_rows_for_stable_toy_data':number_of_rows_for_stable_toy_data,
        'seed_for_the_stable_dataset':seed_for_the_stable_dataset,
    }
