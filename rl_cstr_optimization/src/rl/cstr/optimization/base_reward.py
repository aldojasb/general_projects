"""
Base reward function interface.

This module defines the interface for reward computation:
- compute_reward(): Calculate reward based on state, action, next_state
"""

from abc import ABC, abstractmethod
from typing import Any

class BaseRewardFunction(ABC):
    """
    Abstract base class for defining reward functions in reinforcement learning.

    This class allows for modular, testable, and domain-specific reward design by decoupling
    the reward computation logic from the environment or training loop.

    **Responsibilities:**
    - Compute scalar reward values based on observed transitions.
    - Encapsulate domain logic and performance indicators.
    - Allow easy experimentation with different shaping strategies.

    **Pros:**
    - Makes reward design flexible and extensible.
    - Encourages reusable reward functions across environments or agents.
    - Enables testing or debugging rewards independently from agent behavior.

    **Example Usage:**

        class BiofilmPenaltyReward(BaseRewardFunction):
            def compute(self, state, action, next_state, info):
                biofilm_level = next_state[0]  # assume 0th index = biofilm
                cleaning_cost = action[0]      # assume 0th index = cleaning effort
                reward = - biofilm_level - 0.1 * cleaning_cost
                return reward

    """

    @abstractmethod
    def compute(self, state: Any, action: Any, next_state: Any, info: dict) -> float:
        """
        Compute a scalar reward for a transition in the environment.

        Args:
            state (Any): The current state/observation before the action.
            action (Any): The action taken by the agent.
            next_state (Any): The resulting state/observation after the action.
            info (dict): Auxiliary information returned by the environment (e.g., raw metrics,
                         threshold violations, timeouts, etc.).

        Returns:
            float: A scalar value representing the immediate reward for the transition.
        """
        pass