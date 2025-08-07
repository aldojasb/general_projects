"""
Evaluation script for trained PPO models.

This module provides:
- Model evaluation and testing
- Performance comparison with rule-based baseline
- Visualization of results
"""

import torch
import numpy as np

from typing import Any

import os
import logging
from dotenv import load_dotenv
# Import logger_manager
from logger.manager.logging_config import setup_logging_for_this_script


# ============================================================================
# LOGGING SETUP
# ============================================================================

# Load the .env file only if it exists
dotenv_path = '/workspace/general_projects/rl_cstr_optimization/.env'
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)
    print(f"Loaded environment variables from {dotenv_path}")
else:
    print(f"No .env file found at {dotenv_path}, relying on system environment variables.")

# Access the environment variable, with a fallback
path_to_logs = os.getenv('PATH_TO_SAVE_THE_LOGS')
print(f"Logs will be saved to: {path_to_logs}")
# -

# Set up logging for this script
# This will create logs in the directory specified by PATH_TO_SAVE_THE_LOGS environment variable
setup_logging_for_this_script(log_level=logging.INFO)

# Create a logger for this module
logger = logging.getLogger(__name__)


def load_model(self):
    """Load trained model from checkpoint."""
    pass

def create_rule_based_controller(self):
    """
    Create a simple rule-based controller for comparison.
    
    Rule-based logic:
    - If temperature > 340K: Increase cooling (lower jacket temp)
    - If temperature < 320K: Decrease cooling (higher jacket temp)
    - Otherwise: Maintain current cooling
    """
    def rule_based_action(observation):
        # Denormalize observation if needed
        if observation.max() <= 1.0 and observation.min() >= -1.0:
            o_space = {'low': np.array([0.7, 300, 0.8]), 
                        'high': np.array([1.0, 350, 0.9])}
            temp_real = o_space['low'][1] + (observation[1] + 1) * (o_space['high'][1] - o_space['low'][1]) / 2
        else:
            temp_real = observation[1]
        
        # Rule-based logic
        if temp_real > 340:
            # Too hot - increase cooling
            action = np.array([295.0])  # Lower jacket temperature
        elif temp_real < 320:
            # Too cold - decrease cooling
            action = np.array([300.0])  # Higher jacket temperature
        else:
            # Good temperature - maintain
            action = np.array([297.5])  # Moderate cooling
        
        return action
    
    return rule_based_action

def evaluate_episode(self, 
                    controller: Any,
                    controller_name: str,
                    max_steps: int = 30) -> dict[str, Any]:
    """
    Evaluate a single episode with given controller.
    
    Args:
        controller: Controller function or model
        controller_name: Name for logging
        max_steps: Maximum steps per episode
        
    Returns:
        Dictionary with evaluation results
    """
    pass

def compare_controllers(self, num_episodes: int = 5) -> Dict[str, Any]:
    """
    Compare RL agent vs rule-based controller.
    
    Args:
        num_episodes: Number of episodes to evaluate each controller
        
    Returns:
        Comparison results
    """
    pass

def visualize_episode(self, episode_result: dict[str, Any]):
    """
    Create visualizations for a single episode.
    
    Args:
        episode_result: Results from evaluate_episode
    """
    pass

def main():
    """Main evaluation function."""
    # Check if model exists
    model_path = "model_checkpoints/best_model.pth"
    
    if not os.path.exists(model_path):
        logger.error(f"Model not found: {model_path}")
        logger.error("Please run training first or specify correct model path.")
        return

    # Run comparison
    comparison = compare_controllers(num_episodes=3)
    
    # Visualize best RL episode
    
    logger.info(f"\nEvaluation completed successfully!")


if __name__ == "__main__":
    main() 