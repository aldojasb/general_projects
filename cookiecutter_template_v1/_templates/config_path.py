import os
import argparse

def get_config_path():
    # Check if the environment variable is set
    env_path = os.getenv('PATH_TO_THE_CONFIGURATION_FILE')
    
    if env_path:
        return env_path
    
    # If not, parse the command-line arguments
    parser = argparse.ArgumentParser(description='Provide the path to the configuration file.')
    parser.add_argument('--config', type=str, help='Path to the configuration file')
    args = parser.parse_args()
    
    if args.config:
        return args.config
    else:
        raise ValueError("Configuration file path must be provided"
                         "either as an environment variable 'PATH_TO_THE_CONFIGURATION_FILE'"
                         "or as a command-line argument '--config'.")