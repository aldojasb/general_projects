"""
Base agent interface for reinforcement learning agents.

This module defines the interface that all agents must implement:
- select_action(): Choose action based on current state
- train(): Perform training step
- save(): Save agent model/weights
"""

from abc import ABC, abstractmethod
from typing import Any

class BaseAgent(ABC):
    """
    Abstract base class defining the interface for any reinforcement learning agent 
    (e.g., PPO, DDPG, SAC).

    This interface separates decision-making (action selection), training logic, 
    and persistence (saving/loading). It enables modular experimentation with 
    different agent architectures and training strategies.

    **Responsibilities:**
    - Select actions based on the current state
    - Learn from experience batches
    - Save/load model parameters to/from disk

    **Pros:**
    - Makes it easy to swap in new algorithms without affecting other modules
    - Promotes separation of concerns: decouple agent behavior from training loop
    - Facilitates integration with off-policy or on-policy pipelines

    **Example Usage:**

        class PPOAgent(BaseAgent):
            def select_action(self, state, deterministic=False):
                action = self.policy_network(state)
                return action.argmax() if deterministic else sample(action)

            def train(self, experience_batch):
                self.optimizer.step(experience_batch)

            def save(self, path):
                torch.save(self.policy_network.state_dict(), path)

            def load(self, path):
                self.policy_network.load_state_dict(torch.load(path))

    """

    @abstractmethod
    def select_action(self, state: Any, deterministic: bool = False) -> Any:
        """
        Select an action given the current state.

        Args:
            state (Any): The current state observation from the environment.
                         Can be a NumPy array, tensor, or custom dict depending on setup.
            deterministic (bool, optional): Whether to select the most likely action 
                                            (exploitation) or sample from the policy 
                                            (exploration). Defaults to False.

        Returns:
            Any: The chosen action in agent's output format (can be adapted later).
        """
        pass

    @abstractmethod
    def train(self, experience_batch: Any) -> None:
        """
        Train the agent using a batch of collected experience.

        Args:
            experience_batch (Any): Typically a collection of (state, action, reward, 
                                    next_state, done) tuples, formatted as tensors, 
                                    replay buffers, or dictionaries.

        Returns:
            None
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """
        Persist the agent’s model or policy network to disk.

        Args:
            path (str): Full path to save the model file (e.g., .pt, .pkl, .ckpt).

        Returns:
            None
        """
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """
        Load a saved model or policy from disk.

        Args:
            path (str): Full path to a previously saved model file.

        Returns:
            None
        """
        pass