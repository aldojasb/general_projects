from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional
import numpy as np
from pcgym.envs import BiofilmEnv

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

@dataclass
class BiofilmEnvWrapper(BaseEnvWrapper):
    """
    Concrete implementation of BaseEnvWrapper for the PC-Gym Biofilm environment.

    Responsibilities:
    - Initialize and wrap the raw biofilm environment.
    - Translate between agent actions and environment API.
    - Track the last action to allow passive transitions (e.g., repeating actions).
    
    Example:
        >>> env = BiofilmEnvWrapper(config={"time_horizon": 100})
        >>> state = env.reset()
        >>> next_state, reward, done, info = env.step([10.0, 15.0, 0.3, 0.4, 0.2])
    """

    def __init__(self, config: dict = None):
        """
        Initialize and configure the environment.

        Args:
            config (dict, optional): Configuration dictionary for the biofilm environment.
        """
        self.config = config or {}
        self.env = BiofilmEnv(**self.config)
        self.last_action = None

    def reset(self) -> Any:
        """
        Reset the environment and return the initial observation.

        Returns:
            Any: Initial observation from the environment.
        """
        self.last_action = None
        obs, _ = self.env.reset()
        return obs

    def step(self, action: Any) -> tuple[Any, float, bool, dict]:
        """
        Apply an action to the environment or repeat the last action.

        Args:
            action (Any): Action to apply. If None, repeats last action.

        Returns:
            Tuple[Any, float, bool, dict]: next_state, reward, done, and info dict.
        """
        if action is not None:
            self.last_action = action

        if self.last_action is None:
            raise ValueError("No action has been provided or stored for stepping.")

        next_obs, reward, terminated, truncated, info = self.env.step(self.last_action)
        done = terminated or truncated
        return next_obs, reward, done, info