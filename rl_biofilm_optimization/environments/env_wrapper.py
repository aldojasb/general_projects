from abc import ABC, abstractmethod
from typing import Any, Optional

class BaseEnvWrapper(ABC):
    """
    General interface for wrapping a reinforcement learning environment.

    This abstract base class introduces a lifecycle pattern with three main responsibilities:
    - `initialize()` sets up the environment before training begins.
    - `reset()` begins a new episode and returns the initial observation.
    - `step()` executes a new action or repeats the last one, returning the next state, reward, done flag, and info.

    This structure is ideal for environments where continuous actions are maintained until explicitly changed, 
    such as in industrial control systems (e.g., water treatment plants, chemical reactors).

    ## Pros:
    - Promotes separation of initialization logic from episodic reset logic.
    - Supports more realistic behavior via action persistence.
    - Clean abstraction for plugging into training loops or unit tests.

    ## Cons:
    - Slightly deviates from standard Gym API expectations.
    - Requires careful implementation of internal action memory.

    ## Example:
    ```python
    class MyEnvWrapper(BaseEnvWrapper):
        def __init__(self):
            self.env = SomeGymEnv()
            self.last_action = None

        def initialize(self, config: dict = None, seed: int = None):
            if seed:
                self.env.seed(seed)
            if config:
                self.env.configure(config)

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
    def initialize(self, config: Optional[dict] = None, seed: Optional[int] = None) -> None:
        """
        One-time setup before training begins.

        Args:
            config (Optional[dict]): Optional configuration dictionary for customizing the environment.
            seed (Optional[int]): Optional random seed for reproducibility.
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

    @abstractmethod
    def step(self, action: Optional[Any] = None) -> tuple[Any, float, bool, dict]:
        """
        Execute an action in the environment. If no action is passed, repeat the last action.

        Args:
            action (Optional[Any]): The action to apply. If None, repeat the last applied action.

        Returns:
            Tuple containing:
                - next_state (Any): The resulting observation.
                - reward (float): The reward received after the transition.
                - done (bool): Whether the episode has terminated.
                - info (dict): Additional debug or domain-specific information.
        """
        pass