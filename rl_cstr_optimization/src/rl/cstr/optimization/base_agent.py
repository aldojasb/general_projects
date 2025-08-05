"""
Base agent interface for reinforcement learning agents.

This module defines the interface that all agents must implement:
- select_action(): Choose action based on current state
- train(): Perform training step
- save(): Save agent model/weights
"""

from abc import ABC, abstractmethod
from typing import Any
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


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




# --- Step 1: Define Environment & Actor-Critic Networks --- #

# Assume you have a Gym-like environment ready for CSTR
env = gym.make("CSTR-v0")

state_dim = env.observation_space.shape[0]  # [Ca, Cb, T]
action_dim = env.action_space.shape[0]      # cooling jacket temp adjustment

class ActorCriticNet(nn.Module):
    """
    Actor-Critic Neural Network for PPO (Proximal Policy Optimization).
    
    This network implements the core architecture for PPO algorithms, combining:
    - **Actor**: Outputs action distribution parameters (mean and std) for policy decisions
    - **Critic**: Estimates state values V(s) for advantage computation
    
    **Architecture Design:**
    - Actor: state_dim → 64 → 64 → action_dim (with separate std parameter)
    - Critic: state_dim → 64 → 64 → 1 (state value)
    
    **Key Features:**
    - Uses Tanh activations for bounded, stable outputs
    - Implements Gaussian policy for continuous action spaces
    - Log_std parameter for numerical stability during training
    - Separate actor/critic networks for clear separation of concerns
    
    **Example Usage:**
        >>> model = ActorCriticNet(state_dim=3, action_dim=1)  # CSTR: [Ca, Cb, T] → [cooling_temp]
        >>> state = torch.tensor([[0.8, 0.2, 350.0]])  # [concentration_A, concentration_B, temperature]
        >>> mean, std, value = model(state)
        >>> # mean: target cooling temperature adjustment
        >>> # std: exploration/uncertainty in action selection
        >>> # value: expected future reward from current state
    
    Args:
        state_dim (int): Dimension of state space (e.g., 3 for CSTR: [Ca, Cb, T])
        action_dim (int): Dimension of action space (e.g., 1 for cooling jacket temperature)
    """
    
    def __init__(self, state_dim, action_dim):
        super().__init__()
        
        # ===== ACTOR NETWORK =====
        # The actor learns the policy π(a|s) - how to select actions given states
        # Outputs parameters of a Gaussian distribution for continuous actions
        
        # Shared feature extraction layers
        # Pattern: Linear → Tanh → Linear → Tanh
        # - Linear: Matrix multiplication + bias (learnable transformation)
        # - Tanh: Non-linear activation, bounds outputs to [-1, 1] for stability
        # - 64 hidden units: Empirical choice balancing expressiveness vs efficiency
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64), nn.Tanh(),  # Project state to 64 features
            nn.Linear(64, 64), nn.Tanh(),         # Process features further
        )
        
        # Actor output heads for Gaussian policy parameters
        # mean_head: Outputs μ (mean) of action distribution
        # - Input: 64-dimensional features from actor layers
        # - Output: action_dim values (e.g., 1 for CSTR cooling temperature)
        self.mean_head = nn.Linear(64, action_dim)
        
        # log_std: Learnable parameter for log standard deviation
        # - Why log_std? Numerical stability - can be negative, prevents std=0
        # - Initialized to zeros: exp(0) = 1, so initial std = 1 (reasonable start)
        # - Shape: [action_dim] - one std per action dimension
        # - Bounded in forward() to prevent extreme values during training
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
        # ===== CRITIC NETWORK =====
        # The critic learns V(s) - how good each state is (expected future reward)
        # Used for computing advantages: A(s,a) = Q(s,a) - V(s)
        
        # Critic architecture: deeper than actor for accurate value estimation
        # Pattern: Linear → Tanh → Linear → Tanh → Linear
        # - First two layers: Same as actor for consistent feature processing
        # - Third layer: Projects to single value (state value)
        # - No final activation: Value estimates can be any real number
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64), nn.Tanh(),  # Same feature extraction as actor
            nn.Linear(64, 64), nn.Tanh(),         # Additional processing layer
            nn.Linear(64, 1)                      # Output single state value
        )

    def forward(self, state):
        """
        Forward pass through the actor-critic network.
        
        **Actor Path:**
        1. Extract features from state using shared actor layers
        2. Compute action mean (μ) using mean_head
        3. Compute action std (σ) from log_std parameter
        
        **Critic Path:**
        1. Process state through critic layers
        2. Output state value V(s)
        
        Args:
            state (torch.Tensor): Input state tensor of shape [batch_size, state_dim]
                                 For CSTR: [batch_size, 3] where 3 = [Ca, Cb, T]
        
        Returns:
            tuple: (action_mean, action_std, state_value)
                - action_mean (torch.Tensor): Mean of action distribution [batch_size, action_dim]
                - action_std (torch.Tensor): Standard deviation of action distribution [action_dim]
                - state_value (torch.Tensor): Estimated state value [batch_size, 1]
        
        **Example:**
            >>> state = torch.tensor([[0.8, 0.2, 350.0]])  # CSTR state
            >>> mean, std, value = model(state)
            >>> # mean: [[2.5]] - suggests increase cooling by 2.5K
            >>> # std: [1.2] - exploration range of ±1.2K
            >>> # value: [[15.3]] - good state, expect +15.3 future reward
        """
        
        # ===== ACTOR FORWARD PASS =====
        # Process state through actor layers to extract features
        # Input: [batch_size, state_dim] → Output: [batch_size, 64]
        actor_features = self.actor(state)
        
        # Compute action mean (μ) - the "target" action the policy suggests
        # Input: [batch_size, 64] → Output: [batch_size, action_dim]
        # For CSTR: This is the suggested cooling temperature adjustment
        action_mean = self.mean_head(actor_features)
        
        # Compute action standard deviation (σ) for exploration
        # log_std.clamp(-20, 2): Prevents extreme values during training
        #   - clamp(-20, 2): log_std ∈ [-20, 2]
        #   - exp(): Converts to std ∈ [exp(-20) ≈ 0, exp(2) ≈ 7.4]
        #   - exp(-20) ≈ 0.000000002: Very small exploration (nearly deterministic)
        #   - exp(2) ≈ 7.4: Reasonable exploration range
        # Shape: [action_dim] - same std for all samples in batch
        action_std = self.log_std.clamp(-20, 2).exp()
        
        # ===== CRITIC FORWARD PASS =====
        # Process state through critic to estimate state value V(s)
        # Input: [batch_size, state_dim] → Output: [batch_size, 1]
        # Positive value: Good state (high expected future reward)
        # Negative value: Poor state (low expected future reward)
        # Zero value: Neutral state
        state_value = self.critic(state)
        
        return action_mean, action_std, state_value

# ===== OPTIMIZER INITIALIZATION =====
# We use TWO separate optimizers for actor and critic networks
# This is a key design decision in PPO for training stability and control

# Initialize the actor-critic network
model = ActorCriticNet(state_dim, action_dim)

# ===== ACTOR OPTIMIZER =====
# Optimizes the policy network (actor) parameters
# - model.actor.parameters(): Shared feature extraction layers
# - model.mean_head.weight/bias: Action mean prediction layer
# - model.log_std: Learnable standard deviation parameter
# - Learning rate: 3e-4 (typical for policy optimization)
# 
# Why separate actor optimizer?
# 1. **Different learning objectives**: Actor learns policy, critic learns value function
# 2. **Different learning rates**: Actor often needs slower learning for stability
# 3. **Independent updates**: Prevents one network from interfering with the other
# 4. **Gradient control**: Can apply different gradient clipping/regularization
actor_optimizer = optim.Adam([
    {'params': model.actor.parameters()},  # Shared actor layers
    {'params': [model.mean_head.weight, model.mean_head.bias, model.log_std]}  # Policy output parameters
], lr=3e-4)

# ===== CRITIC OPTIMIZER =====
# Optimizes the value function network (critic) parameters
# - model.critic.parameters(): All critic network layers
# - Learning rate: 1e-3 (faster than actor for accurate value estimation)
#
# Why separate critic optimizer?
# 1. **Faster convergence**: Value functions often converge faster than policies
# 2. **Different loss functions**: MSE for critic vs. policy gradient for actor
# 3. **Stability**: Prevents critic updates from destabilizing policy learning
# 4. **Independent momentum**: Adam's momentum states are separate for each network
critic_optimizer = optim.Adam(model.critic.parameters(), lr=1e-3)


# --- Step 2: Experience Collection (Rollouts) --- #

def collect_trajectories(model, env, steps=2048):
    state = env.reset()
    states, actions, rewards, dones, values, log_probs = [], [], [], [], [], []
    for _ in range(steps):
        state_tensor = torch.FloatTensor(state)
        with torch.no_grad():
            mean, std, value = model(state_tensor)
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)

        next_state, reward, done, _ = env.step(action.numpy())

        # Collect experience
        states.append(state)
        actions.append(action.numpy())
        rewards.append(reward)
        dones.append(done)
        values.append(value.item())
        log_probs.append(log_prob.item())

        state = next_state
        if done:
            state = env.reset()

    # Convert to tensors
    return (states, actions, rewards, dones, values, log_probs)


# --- Step 3: Compute Advantages (GAE) --- #

def compute_gae(rewards, dones, values, gamma=0.99, lam=0.95):
    advantages = np.zeros_like(rewards)
    last_gae = 0
    values = np.array(values + [0])  # bootstrap next value

    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
        advantages[t] = last_gae = delta + gamma * lam * (1 - dones[t]) * last_gae

    returns = advantages + values[:-1]
    # Normalization (critical!)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return torch.FloatTensor(advantages), torch.FloatTensor(returns)


# --- Step 4 & 5: PPO Update (Clipped Objective & Value Loss) --- #

def ppo_update(model, states, actions, log_probs_old, returns, advantages, clip=0.2, epochs=10):
    states = torch.FloatTensor(states)
    actions = torch.FloatTensor(actions)
    log_probs_old = torch.FloatTensor(log_probs_old)

    for _ in range(epochs):
        mean, std, values = model(states)
        dist = torch.distributions.Normal(mean, std)

        log_probs_new = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().mean()

        # Clipped surrogate objective
        ratios = torch.exp(log_probs_new - log_probs_old)
        surrogate1 = ratios * advantages
        surrogate2 = torch.clamp(ratios, 1 - clip, 1 + clip) * advantages
        actor_loss = -torch.min(surrogate1, surrogate2).mean()

        # Critic loss (MSE)
        critic_loss = F.mse_loss(values.squeeze(), returns)

        # Total loss with entropy bonus
        loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

        # ===== ACTOR UPDATE =====
        # Update the policy network (actor) using the clipped surrogate objective
        # This is the core of PPO - improving the policy while preventing large changes
        
        # Clear previous gradients for actor parameters
        actor_optimizer.zero_grad()
        
        # Compute gradients for actor loss (policy improvement)
        # retain_graph=True: Keep computational graph for critic update
        actor_loss.backward(retain_graph=True)
        
        # Gradient clipping: Prevent exploding gradients that could destabilize training
        # Clips gradient norm to 0.5 for all model parameters (both actor and critic)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        
        # Apply the gradient updates to actor parameters only
        # This updates: actor layers, mean_head, and log_std
        actor_optimizer.step()

        # ===== CRITIC UPDATE =====
        # Update the value function network (critic) using MSE loss
        # This improves the accuracy of state value estimates
        
        # Clear previous gradients for critic parameters
        critic_optimizer.zero_grad()
        
        # Compute gradients for critic loss (value function improvement)
        # No retain_graph needed since this is the final backward pass
        critic_loss.backward()
        
        # Apply the gradient updates to critic parameters only
        # This updates: all critic network layers
        critic_optimizer.step()


# --- Main training loop --- #
num_updates = 1000
for update in range(num_updates):
    # Collect rollouts
    states, actions, rewards, dones, values, log_probs_old = collect_trajectories(model, env)

    # Compute GAE advantages
    advantages, returns = compute_gae(rewards, dones, values)

    # Update PPO policy and value network
    ppo_update(model, states, actions, log_probs_old, returns, advantages)

    if (update + 1) % 100 == 0:
        print(f"Update {update + 1}/{num_updates} completed.")

# Save trained Actor-Critic model after training
torch.save(model.state_dict(), "ppo_actor_critic_cstr.pth")  # clearly storing both actor and critic parameters
