import os
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
import json

# Get the logger for this module
import logging
from database_generator.logging_configuration import setup_logging_for_this_script
setup_logging_for_this_script()
# Get the logger for this module
logger = logging.getLogger(__name__)

def get_config_path():
    # Check if the environment variable is set
    env_path = os.getenv('PATH_TO_THE_CONFIGURATION_FILE')
    
    if env_path:
        return env_path
    else:
        logger.error("Configuration file path must be provided"
                         " either as an environment variable 'PATH_TO_THE_CONFIGURATION_FILE'")
        raise ValueError("Configuration file path must be provided"
                         " either as an environment variable 'PATH_TO_THE_CONFIGURATION_FILE'")


# Parse the timestamps with error handling
def parse_timestamp(timestamp_str: str) -> datetime:
    try:
        # Replace '-00' with '+00:00' to make it a valid ISO 8601 format
        if timestamp_str.endswith('-00'):
            timestamp_str = timestamp_str.replace('-00', '+00:00')
        return datetime.fromisoformat(timestamp_str)
    except ValueError as e:
        logger.error(f"Error parsing timestamp: {timestamp_str}. {e}"
                        "datetime should be in ISO 8601 format")
        raise


def load_and_process_params(file_path: str) -> dict:
    """
    Load and process parameters from a JSON file, returning a flat dictionary with all parameter names and values.

    Parameters:
    ----------
    file_path : str
        The path to the JSON file containing the parameters.

    Returns:
    -------
    dict
        A dictionary containing all parameters with their names as keys and their respective values.
    """
    
    def recursive_extract(params: dict, result: dict = None) -> dict:
        """
        Recursively extract all key-value pairs from a nested dictionary without parent keys.

        Parameters:
        ----------
        params : dict
            The dictionary to extract key-value pairs from.
        result : dict
            The dictionary to store the flattened parameters.

        Returns:
        -------
        dict
            A dictionary containing all flattened parameters without parent keys.
        """
        if result is None:
            result = {}

        for key, value in params.items():
            if isinstance(value, dict):
                # Recursively process nested dictionaries without adding parent keys
                recursive_extract(value, result)
            else:
                # Use the current key without any parent key
                result[key] = value
                logger.info(f'{key} = {value}')

        return result


    # Load parameters from JSON file
    with open(file_path, "r") as file:
        params = json.load(file)

    # Recursively extract all parameters
    processed_params = recursive_extract(params)

    return processed_params
