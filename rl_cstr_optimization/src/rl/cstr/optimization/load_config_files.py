"""
Configuration loader for CSTR environment parameters.

This module provides functions to load and process configuration files
for the CSTR (Continuously Stirred Tank Reactor) environment.
"""

import yaml
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional


def load_cstr_config(config_path: str) -> Dict[str, Any]:
    """
    Load CSTR environment configuration from YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Dictionary containing the processed configuration
        
    Raises:
        FileNotFoundError: If the config file doesn't exist
        yaml.YAMLError: If the YAML file is malformed
        ValueError: If the configuration is invalid
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML file {config_path}: {e}")
    
    # Validate and process the configuration
    config = _validate_and_process_config(config)
    
    return config


def _validate_and_process_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and process the loaded configuration.
    
    Args:
        config: Raw configuration dictionary
        
    Returns:
        Processed configuration dictionary
        
    Raises:
        ValueError: If required fields are missing or invalid
    """
    # Check required sections exist
    required_sections = ['simulation', 'action_space', 'observation_space', 
                        'initial_conditions', 'setpoints', 'reward_scaling', 'environment']
    
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required section: {section}")
    
    # Process setpoints to match the expected format
    nsteps = config['simulation']['nsteps']
    sp_values = _process_setpoints(config['setpoints']['ca_profile'], nsteps)
    
    # Create the processed configuration
    processed_config = {
        'simulation': config['simulation'],
        'action_space': config['action_space'],
        'observation_space': config['observation_space'],
        'initial_conditions': config['initial_conditions'],
        'setpoints': {
            'Ca': sp_values
        },
        'reward_scaling': config['reward_scaling'],
        'environment': config['environment']
    }
    
    return processed_config


def _process_setpoints(ca_profile: list, nsteps: int) -> list:
    """
    Process setpoint profile to create a list of exactly nsteps values.
    
    This function takes a list of setpoint phases (each with a value and duration)
    and converts it into a flat list of setpoint values that matches the required
    number of steps. It handles cases where the total duration doesn't exactly
    match nsteps by either truncating or extending the list.
    
    Example:
        Input ca_profile: [
            {'value': 0.85, 'duration': 3},  # 0.85 for 3 steps
            {'value': 0.9, 'duration': 3},   # 0.9 for 3 steps  
            {'value': 0.87, 'duration': 4}   # 0.87 for 4 steps
        ]
        Input nsteps: 10
        
        Output: [0.85, 0.85, 0.85, 0.9, 0.9, 0.9, 0.87, 0.87, 0.87, 0.87]
    
    Args:
        ca_profile: List of dictionaries, each containing:
            - 'value': float - The setpoint value for this phase
            - 'duration': int - Number of steps this value should be applied
        nsteps: Total number of steps required for the simulation
        
    Returns:
        List of float values with exactly nsteps elements, representing the
        setpoint values for each step of the simulation
        
    Raises:
        ValueError: If ca_profile is empty or contains invalid data
    """
    # Validate input
    if not ca_profile:
        raise ValueError("ca_profile cannot be empty")
    
    # Step 1: Convert phase-based profile to flat list of values
    # Each phase specifies a value and how long to apply it
    sp_values = []
    
    for phase in ca_profile:
        value = phase['value']
        duration = phase['duration']
        
        # Extend the list with 'duration' copies of the current value
        # This creates a flat sequence like: [0.85, 0.85, 0.85, 0.9, 0.9, 0.9, ...]
        sp_values.extend([value] * duration)
    
    # Step 2: Ensure the list has exactly nsteps elements
    # This handles cases where the total duration doesn't match nsteps exactly
    if len(sp_values) != nsteps:
        if len(sp_values) > nsteps:
            # Case A: We have too many values (e.g., 12 values for 10 steps)
            # Solution: Truncate to keep only the first nsteps values
            sp_values = sp_values[:nsteps]
            
        else:
            # Case B: We have too few values (e.g., 8 values for 10 steps)  
            # Solution: Extend with copies of the last value
            # This assumes the last setpoint should continue for remaining steps
            sp_values.extend([sp_values[-1]] * (nsteps - len(sp_values)))
    
    return sp_values


def config_to_env_params(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert configuration dictionary to pc-gym environment parameters format.
    
    Args:
        config: Processed configuration dictionary
        
    Returns:
        Dictionary in the format expected by pc-gym make_env function
    """
    return {
        'N': config['simulation']['nsteps'],
        'tsim': config['simulation']['time_hours'],
        'SP': config['setpoints'],
        'o_space': {
            'low': np.array(config['observation_space']['low']),
            'high': np.array(config['observation_space']['high'])
        },
        'a_space': {
            'low': np.array(config['action_space']['low']),
            'high': np.array(config['action_space']['high'])
        },
        'x0': np.array(config['initial_conditions']['x0']),
        'r_scale': config['reward_scaling'],
        'model': config['environment']['model'],
        'normalise_a': config['environment']['normalise_a'],
        'normalise_o': config['environment']['normalise_o'],
        'noise': config['environment']['noise'],
        'integration_method': config['environment']['integration_method'],
        'noise_percentage': config['environment']['noise_percentage'],
    }


def load_and_create_env_params(config_path: str) -> Dict[str, Any]:
    """
    Convenience function to load config and convert to environment parameters.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Dictionary ready to use with pc-gym make_env function
    """
    config = load_cstr_config(config_path)
    return config_to_env_params(config)
