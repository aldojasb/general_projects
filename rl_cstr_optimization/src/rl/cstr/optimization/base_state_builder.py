"""
Base state builder interface.

This module defines the interface for state representation:
- build_state(): Transform observation into state representation
- normalize_observations(): Convert real observations to normalized range
- denormalize_observations(): Convert normalized observations to real values
"""

from abc import ABC, abstractmethod
from typing import Any, Union
import numpy as np

class BaseStateBuilder(ABC):
    """
    Abstract base class for transforming raw observations into RL agent-friendly states.

    This class supports modular feature engineering by cleanly separating the raw observations
    from the internal state representation used by the agent. It allows you to experiment with
    different state encodings (e.g., stacked observations, moving averages, normalized values).

    **Responsibilities:**
    - Convert environment observations into agent-consumable state vectors or tensors.
    - Optionally apply normalization, filtering, or feature extraction.
    - Standardize inputs across environments or observation formats.

    **Pros:**
    - Facilitates feature engineering and ablation studies.
    - Makes it easier to debug, test, or swap in new state representations.
    - Keeps training and environment code clean and focused.

    **Example Usage:**

        class Last5MeanStateBuilder(BaseStateBuilder):
            def __init__(self):
                self.last_n = []

            def build_state(self, observation):
                self.last_n.append(observation)
                if len(self.last_n) > 5:
                    self.last_n.pop(0)
                return sum(self.last_n) / len(self.last_n)

    """

    @abstractmethod
    def build_state(self, observation: Any) -> Any:
        """
        Convert a raw observation from the environment into an agent-readable state.

        Args:
            observation (Any): The raw data returned by the environment at each step.
                               This could be a dict, array, or any structured object.

        Returns:
            Any: The processed state (e.g., a float list, numpy array, tensor)
                 that will be used by the RL agent for decision-making.
        """
        pass


def normalize_observations(observations: Union[np.ndarray, list[np.ndarray]], 
                         observation_bounds: dict) -> np.ndarray:
    """
    Normalize observations from real-world values to [-1, 1] range.
    
    This function converts observations from their real-world physical units
    to a normalized range that's suitable for neural networks and RL algorithms.
    
    Args:
        observations (Union[np.ndarray, List[np.ndarray]]): 
            - Single observation array with shape (n_features,)
            - List of observation arrays
            - Array of observations with shape (n_observations, n_features)
        observation_bounds (dict): Dictionary with 'low' and 'high' bounds for real values
            Example: {'low': np.array([0.7, 300, 0.8]), 'high': np.array([1.0, 350, 0.9])}
    
    Returns:
        np.ndarray: Normalized observations in [-1, 1] range
        
    Example:
        >>> obs = np.array([0.8, 330, 0.85])
        >>> bounds = {'low': np.array([0.7, 300, 0.8]), 'high': np.array([1.0, 350, 0.9])}
        >>> norm_obs = normalize_observations(obs, bounds)
        >>> print(norm_obs)  # [-0.33, 0.2, 0.5] - normalized values
    """
    observations = np.array(observations)
    
    # Handle single observation vs batch
    if observations.ndim == 1:
        observations = observations.reshape(1, -1)
    
    low = observation_bounds['low']
    high = observation_bounds['high']
    
    # Normalize to [0, 1] range first
    normalized_01 = (observations - low) / (high - low)
    
    # Then convert to [-1, 1] range
    normalized = 2 * normalized_01 - 1
    
    # Clip to ensure values are within bounds
    normalized = np.clip(normalized, -1, 1)
    
    return normalized.squeeze() if observations.shape[0] == 1 else normalized


def denormalize_observations(normalized_observations: Union[np.ndarray, list[np.ndarray]], 
                           observation_bounds: dict) -> np.ndarray:
    """
    Denormalize observations from [-1, 1] range to real-world values.
    
    This function converts normalized observations back to their real-world
    physical units for interpretation and visualization.
    
    Args:
        normalized_observations (Union[np.ndarray, List[np.ndarray]]): 
            - Single normalized observation array with shape (n_features,)
            - List of normalized observation arrays
            - Array of normalized observations with shape (n_observations, n_features)
        observation_bounds (dict): Dictionary with 'low' and 'high' bounds for real values
            Example: {'low': np.array([0.7, 300, 0.8]), 'high': np.array([1.0, 350, 0.9])}
    
    Returns:
        np.ndarray: Denormalized observations in real-world units
        
    Example:
        >>> norm_obs = np.array([-0.33, 0.2, 0.5])
        >>> bounds = {'low': np.array([0.7, 300, 0.8]), 'high': np.array([1.0, 350, 0.9])}
        >>> real_obs = denormalize_observations(norm_obs, bounds)
        >>> print(real_obs)  # [0.8, 330, 0.85] - real values
    """
    normalized_observations = np.array(normalized_observations)
    
    # Handle single observation vs batch
    if normalized_observations.ndim == 1:
        normalized_observations = normalized_observations.reshape(1, -1)
    
    low = observation_bounds['low']
    high = observation_bounds['high']
    
    # Convert from [-1, 1] to [0, 1] range
    normalized_01 = (normalized_observations + 1) / 2
    
    # Then scale to real bounds
    denormalized = low + normalized_01 * (high - low)
    
    return denormalized.squeeze() if normalized_observations.shape[0] == 1 else denormalized

