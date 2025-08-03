"""
Base state builder interface.

This module defines the interface for state representation:
- build_state(): Transform observation into state representation
"""

from abc import ABC, abstractmethod
from typing import Any

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