import os
import argparse
import sys

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
