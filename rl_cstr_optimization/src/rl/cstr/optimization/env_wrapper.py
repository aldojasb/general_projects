from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional
import numpy as np
from pcgym import make_env
from rl.cstr.optimization.visualization import (
    plot_state_variables,
    plot_control_actions,
    plot_reward_evolution)
from rl.cstr.optimization.load_config_files import load_and_create_env_params
from rl.cstr.optimization.base_state_builder import denormalize_observations
from rl.cstr.optimization.base_action_adapter import denormalize_actions

# ===============================
# Abstract Base Class
# ===============================

class BaseEnvWrapper(ABC):
    """
    General interface for wrapping a reinforcement learning environment.

    This abstract base class has the following main responsibilities:
    - `step()` executes a new action or repeats the last one, returning the next state, reward, done flag, and info.
    - `reset()` begins a new episode and returns the initial observation.

    This structure is ideal for environments where continuous actions are maintained until explicitly changed, 
    such as in industrial control systems (e.g., water treatment plants, chemical reactors).

    ## Pros:
    - Supports more realistic behavior via action persistence.
    - Clean abstraction for plugging into training loops or unit tests.

    ## Example:
    ```python
    class MyEnvWrapper(BaseEnvWrapper):
        def __init__(self):
            self.env = SomeGymEnv()
            self.last_action = None

        def reset(self):
            obs = self.env.reset()
            self.last_action = None
            return obs

        def step(self, action: Optional[Any] = None):
            if action is not None:
                self.last_action = action
            action_to_apply = self.last_action
            return self.env.step(action_to_apply)
    ```

    """

    @abstractmethod
    def step(self, action: Optional[Any] = None) -> tuple[Any, bool]:
        """
        Execute an action in the environment. If no action is passed, repeat the last action.

        Args:
            action (Optional[Any]): The action to apply. If None, repeat the last applied action.

        Returns:
            Tuple containing:
                - next_state (Any): The resulting observation.
                - done (bool): Whether the episode has terminated.
        """
        pass


    @abstractmethod
    def reset(self) -> Any:
        """
        Reset the environment and return the initial observation for a new episode.

        Returns:
            Any: The initial observation after reset.
        """
        pass

# ===============================
# Concrete Implementation
# ===============================

# TODO: Implement concrete implementation of BaseEnvWrapper

"""
CSTR (Continuously Stirred Tank Reactor) Environment Demo
========================================================

This demo showcases the CSTR environment from pc-gym, which simulates
a continuously stirred tank reactor for chemical process control.

The CSTR is a well-established model that's thoroughly tested and stable,
making it perfect for learning and experimentation.

State Variables (3 total):
- Ca: Concentration of reactant A (mol/L)
- T: Temperature (K)
- Cb: Concentration of reactant B (mol/L)

Action Variables (1 total):
- Tc: Coolant temperature (K)

Observations (3 total):
- Ca, T, Cb: Concentration A, Temperature, Concentration B
"""

import numpy as np
from pcgym import make_env
from rl.cstr.optimization.visualization import (
    plot_state_variables,
    plot_control_actions,
    plot_reward_evolution)
from rl.cstr.optimization.load_config_files import load_and_create_env_params
from rl.cstr.optimization.base_state_builder import denormalize_observations
from rl.cstr.optimization.base_action_adapter import denormalize_actions

# +
# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

# Load configuration from YAML file
config_path = "/workspace/general_projects/rl_cstr_optimization/config/environments/cstr_environment.yaml"
env_params = load_and_create_env_params(config_path)

# Extract action space bounds from env_params for use in the notebook
a_space = env_params['a_space']
o_space = env_params['o_space']
nsteps = env_params['N']

print(f"Configuration loaded: {env_params}")
print(f"Action space: {a_space}")
print(f"Observation space: {o_space}")
print(f"Number of steps: {nsteps}")


# +
# Create the CSTR environment with proper parameters
# The environment simulates a continuously stirred tank reactor for chemical process control
env = make_env(env_params)

# Reset the environment to get initial state
# This returns the initial observation (concentrations at reactor outlet)
initial_observation, initial_info = env.reset()

print("=" * 60)
print(f"CSTR REACTOR DEMO - {nsteps} STEP SIMULATION")
print("=" * 60)
print(f"Initial observation (normalized): {initial_observation}")