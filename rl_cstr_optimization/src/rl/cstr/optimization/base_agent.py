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

    This interface separates training logic, and persistence (saving/loading).
    It enables modular experimentation with different agent architectures and training strategies.

    **Responsibilities:**
    - Train the agent using a batch of collected experience.
    - Save/load model parameters to/from disk

    **Pros:**
    - Makes it easy to swap in new algorithms without affecting other modules
    - Promotes separation of concerns: decouple agent behavior from training loop
    - Facilitates integration with off-policy or on-policy pipelines

    **Example Usage:**

        class PPOAgent(BaseAgent):
            def train(self, experience_batch):
                self.optimizer.step(experience_batch)

            def save(self, path):
                torch.save(self.policy_network.state_dict(), path)

            def load(self, path):
                self.policy_network.load_state_dict(torch.load(path))
    """

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




# # --- Define Environment & Actor-Critic Networks --- #

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
    
    def __init__(self, state_dim: int, action_dim: int) -> None:
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

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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


# # ===== EXPERIENCE COLLECTION (ROLLOUTS) =====
# # This is the data collection phase of PPO - gathering experience to learn from
# # 
# # **Role in PPO Algorithm:**
# # 1. **Policy Evaluation**: Test current policy in environment to see how well it performs
# # 2. **Data Collection**: Gather (state, action, reward, next_state) tuples for training
# # 3. **Value Estimation**: Get critic's value estimates for advantage computation
# # 4. **Policy Sampling**: Record action probabilities for importance sampling in PPO
# #
# # **Why "Rollouts"?** 
# # - We "roll out" the current policy for a fixed number of steps
# # - This creates a trajectory of experience that we'll use to improve the policy
# # - Think of it as "testing" the current policy to see what works and what doesn't

def collect_trajectories(
    model: ActorCriticNet, 
    env: Any, 
    steps: int = 2048
) -> tuple[list[np.ndarray], list[np.ndarray], list[float], list[bool], list[float], list[float]]:
    """
    Collect experience trajectories using the current policy.
    
    **PPO Context:**
    This function implements the "rollout" phase of PPO, where we:
    1. Use the current policy to interact with the environment
    2. Collect experience data for policy improvement
    3. Record action probabilities for importance sampling
    4. Get value estimates for advantage computation
    
    **For CSTR Context:**
    - Collects sequences of reactor control decisions
    - Records temperature adjustments and their outcomes
    - Gathers data on how well the current control policy performs
    
    **What is a Trajectory?**
    A trajectory is a sequence of (state, action, reward) tuples collected by
    following the current policy. Think of it as a "test run" of your policy
    to see how well it performs and gather data for improvement.
    
    **Why 2048 steps?**
    - Standard PPO hyperparameter for experience collection
    - Balances data quality (enough experience) with computational efficiency
    - Large enough to get meaningful statistics, small enough to update frequently
    
    Args:
        model (ActorCriticNet): Current policy and value function
        env: Environment to interact with (CSTR environment)
        steps (int): Number of steps to collect (default: 2048)
    
    Returns:
        tuple: (states, actions, rewards, dones, values, log_probs)
            - states: List of observed states [Ca, Cb, T]
            - actions: List of actions taken (cooling temperature adjustments)
            - rewards: List of rewards received
            - dones: List of episode termination flags
            - values: List of critic's value estimates
            - log_probs: List of action log probabilities (for importance sampling)
    
    **Example Usage:**
        >>> states, actions, rewards, dones, values, log_probs = collect_trajectories(model, env)
        >>> # states: [[0.8, 0.2, 350.0], [0.7, 0.3, 348.0], ...]
        >>> # actions: [[2.3], [-1.7], [0.5], ...]  # temperature adjustments
        >>> # rewards: [15.2, 12.8, 18.1, ...]  # conversion efficiency rewards
    """
    
    # ===== INITIALIZATION =====
    # Reset environment to start fresh episode
    # env.reset(): Returns initial state and resets environment to starting conditions
    # For CSTR: Returns initial reactor conditions [Ca_initial, Cb_initial, T_initial]
    # This ensures we start from a safe, known state for consistent data collection
    state = env.reset()
    
    # Initialize empty lists to store trajectory data
    # These will hold the experience collected during the rollout
    # Each list will grow to length 'steps' by the end of the function
    states, actions, rewards, dones, values, log_probs = [], [], [], [], [], []
    
    # ===== EXPERIENCE COLLECTION LOOP =====
    # Collect experience for 'steps' timesteps
    # Each iteration: observe state → select action → get reward → observe next state
    # This creates a trajectory of experience that we'll use to improve the policy
    for _ in range(steps):
        
        # ===== STATE PROCESSING =====
        # Convert state from numpy array to PyTorch tensor
        # torch.FloatTensor(state): Converts numpy array to tensor for neural network input
        # Required because PyTorch models expect tensor inputs, not numpy arrays
        # For CSTR: Converts [Ca, Cb, T] numpy array to tensor for model input
        state_tensor = torch.FloatTensor(state)
        
        # ===== POLICY INFERENCE (NO GRADIENT COMPUTATION) =====
        # with torch.no_grad(): Disables gradient computation for efficiency
        # During rollout, we only want to collect data, not compute gradients
        # This saves memory and computation since we're not training yet
        # Critical for efficiency: we don't need gradients during data collection
        with torch.no_grad():
            # Forward pass through actor-critic network
            # model(state_tensor): Returns (action_mean, action_std, state_value)
            # - action_mean: Target action the policy suggests (e.g., +2.5K cooling)
            # - action_std: Exploration/uncertainty in action selection (e.g., ±1.2K)
            # - state_value: Critic's estimate of state value V(s) (e.g., 15.3)
            # For CSTR: Predicts optimal cooling temperature adjustment and reactor value
            mean, std, value = model(state_tensor)
            
            # ===== ACTION SAMPLING =====
            # Create probability distribution for action sampling
            # torch.distributions.Normal(mean, std): Creates Gaussian distribution
            # - mean: Center of the distribution (target action)
            # - std: Spread of the distribution (exploration range)
            # For CSTR: Distribution over cooling temperature adjustments
            # Example: mean=2.5, std=1.2 → distribution centered at +2.5 with ±1.2 spread
            dist = torch.distributions.Normal(mean, std)
            
            # Sample action from the distribution
            # dist.sample(): Draws random action from the Gaussian distribution
            # This implements exploration - sometimes actions differ from the mean
            # For CSTR: Actual cooling temperature adjustment (e.g., +2.3, -1.7, +3.1)
            # The sampled action may be different from the mean due to exploration
            action = dist.sample()
            
            # Compute log probability of the sampled action
            # dist.log_prob(action): Log probability of the chosen action
            # .sum(dim=-1): Sum across action dimensions (if multi-dimensional)
            # This is crucial for PPO's importance sampling ratio calculation
            # For CSTR: Log probability of the chosen temperature adjustment
            # Example: If action=2.3 and mean=2.5, std=1.2, log_prob ≈ -0.5
            log_prob = dist.log_prob(action).sum(dim=-1)

        # ===== ENVIRONMENT INTERACTION =====
        # Apply action to environment and observe results
        # env.step(action.numpy()): Executes action and returns (next_state, reward, done, info)
        # action.numpy(): Converts tensor back to numpy for environment compatibility
        # For CSTR: Applies cooling temperature adjustment and observes reactor response
        # Returns: new reactor conditions, reward based on efficiency/safety, episode status
        next_state, reward, done, _ = env.step(action.numpy())

        # ===== EXPERIENCE STORAGE =====
        # Store all collected data for later training
        # Each piece is essential for PPO's policy improvement
        
        # states.append(state): Current state before action
        # For CSTR: [Ca, Cb, T] before temperature adjustment
        # Example: [0.8, 0.2, 350.0] - reactant concentrations and temperature
        states.append(state)
        
        # actions.append(action.numpy()): Action that was taken
        # For CSTR: Cooling temperature adjustment that was applied
        # Example: [2.3] - increased cooling by 2.3K
        actions.append(action.numpy())
        
        # rewards.append(reward): Reward received for this action
        # For CSTR: Reward based on conversion efficiency and safety
        # Example: 15.2 - good reward for maintaining optimal conditions
        rewards.append(reward)
        
        # dones.append(done): Whether episode ended
        # For CSTR: True if reactor reached unsafe conditions or time limit
        # Example: False - episode continues, True - reactor overheated
        dones.append(done)
        
        # values.append(value.item()): Critic's value estimate
        # For CSTR: Expected future reward from current reactor state
        # Example: 15.3 - critic expects good future performance
        values.append(value.item())
        
        # log_probs.append(log_prob.item()): Action probability for importance sampling
        # For CSTR: Log probability of the chosen temperature adjustment
        # Critical for PPO's ratio calculation: π_new(a|s) / π_old(a|s)
        # Example: -0.5 - log probability of the chosen action
        log_probs.append(log_prob.item())

        # ===== STATE TRANSITION =====
        # Move to next state for next iteration
        # For CSTR: Update to new reactor conditions after temperature adjustment
        state = next_state
        
        # ===== EPISODE RESET =====
        # If episode ended, reset environment for fresh start
        # env.reset(): Returns new initial state
        # For CSTR: Resets reactor to safe initial conditions
        # This prevents the agent from getting stuck in bad states
        if done:
            state = env.reset()

    # ===== RETURN COLLECTED EXPERIENCE =====
    # Return all collected data as lists (will be converted to tensors later)
    # This data will be used for:
    # 1. Computing advantages (GAE) - how much better actions were than expected
    # 2. Policy updates (PPO clipped objective) - improve the policy
    # 3. Value function updates (MSE loss) - improve value estimates
    # 
    # For CSTR: Returns trajectory of reactor control decisions and outcomes
    # This data shows how well the current control policy performed
    return (states, actions, rewards, dones, values, log_probs)


# ===== STEP 3: COMPUTE ADVANTAGES (GAE) =====
# Generalized Advantage Estimation (GAE) - a sophisticated way to estimate how much
# better or worse actions were compared to expectations
#
# **What is GAE?**
# GAE provides a stable way to estimate advantages: A(s,a) = Q(s,a) - V(s)
# - Q(s,a): How good was this action? (estimated from rewards)
# - V(s): How good did we expect this state to be? (from critic)
# - A(s,a): How much better/worse was the action than expected?
#
# **Why GAE matters for PPO:**
# 1. **Policy Improvement**: Positive advantages → encourage similar actions
# 2. **Policy Degradation**: Negative advantages → discourage similar actions
# 3. **Stable Training**: GAE reduces variance in advantage estimates
#
# **Analogy: Restaurant Review System**
# - Expected Rating (V(s)): Critics predict restaurant quality (e.g., 7/10)
# - Actual Experience (reward): You visit and rate it (e.g., 9/10)
# - Advantage: Difference between expectation and reality (9 - 7 = +2)
# - GAE: Sophisticated calculation considering future expectations too

def compute_gae(
    rewards: list[float], 
    dones: list[bool], 
    values: list[float], 
    gamma: float = 0.99, 
    lam: float = 0.95
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Generalized Advantage Estimation (GAE) for policy optimization.
    
    **What is GAE?**
    GAE is a method to estimate advantages that balances bias and variance.
    It tells us "how much better or worse was each action compared to expectations?"
    
    **Mathematical Formula:**
    GAE(γ,λ) = Σ(γλ)^l * δ_{t+l}
    where δ_t = r_t + γV(s_{t+1}) - V(s_t)
    
    **For CSTR Context:**
    - rewards: Conversion efficiency and safety rewards from temperature adjustments
    - values: Critic's predictions of expected future rewards
    - advantages: How much better/worse each temperature adjustment was than expected
    - returns: Total expected future rewards from each state
    
    **Key Parameters:**
    - gamma (γ): Discount factor - how much we value future vs immediate rewards
    - lambda (λ): GAE parameter - balances bias vs variance in advantage estimation
    - dones: Episode termination flags - when reactor reaches unsafe conditions
    
    **Why GAE is Critical for PPO:**
    1. **Stable Training**: Reduces variance in advantage estimates
    2. **Policy Guidance**: Positive advantages encourage good actions
    3. **Value Learning**: Helps critic learn accurate value functions
    4. **Exploration Control**: Prevents over-optimization of noisy rewards
    
    Args:
        rewards (list): List of rewards received for each action
        dones (list): List of episode termination flags (True/False)
        values (list): List of critic's value estimates for each state
        gamma (float): Discount factor for future rewards (default: 0.99)
        lam (float): GAE parameter for bias-variance trade-off (default: 0.95)
    
    Returns:
        tuple: (advantages, returns)
            - advantages: Normalized advantage estimates for each action
            - returns: Total expected future rewards from each state
            - raw_advantages: Raw advantage estimates for each action
    
    **Example:**
        >>> rewards = [15.2, 12.8, 18.1, 14.5]  # CSTR conversion rewards
        >>> values = [15.0, 13.0, 17.5, 14.0]   # Critic's predictions
        >>> dones = [False, False, False, True]   # Episode termination
        >>> advantages, returns = compute_gae(rewards, dones, values)
        >>> # advantages: [0.2, -0.2, 0.6, 0.5]  # How much better/worse than expected
        >>> # returns: [15.2, 12.8, 18.1, 14.5]  # Total future rewards
        >>> raw_advantages: [12.0, 14.0, 10.7, 15.5]  # Raw advantage estimates
    """
    
    # ===== INITIALIZATION =====
    # Initialize advantage array with same shape as rewards
    # np.zeros_like(rewards): Creates array of zeros with same shape as rewards
    # This will store our computed advantages for each timestep
    advantages = np.zeros_like(rewards)
    
    # Initialize last_gae to 0 for the final timestep
    # last_gae: The GAE value from the next timestep (used in recursive calculation)
    # Starts at 0 because there's no "next" advantage after the final timestep
    last_gae = 0
    
    # Bootstrap the value function for the final timestep
    # values + [0]: Adds a zero value for the "next" state after the final timestep
    # This is called "bootstrapping" - we assume the final state has zero value
    # For CSTR: After the final timestep, we assume no more rewards (reactor stops)
    values = np.array(values + [0])  # bootstrap next value
    
    # ===== REVERSE ITERATION FOR GAE COMPUTATION =====
    # We compute GAE backwards because it depends on future values
    # This is the core of GAE - working backwards from the end of the trajectory
    # For CSTR: We start from the final temperature adjustment and work backwards
    for t in reversed(range(len(rewards))):
        
        # ===== COMPUTE DELTA (IMMEDIATE ADVANTAGE) =====
        # delta = r_t + γV(s_{t+1}) - V(s_t)
        # This is the immediate difference between actual and expected performance
        
        # rewards[t]: Actual reward received at timestep t
        # For CSTR: Actual conversion efficiency reward from temperature adjustment
        
        # gamma * values[t + 1] * (1 - dones[t]): Discounted future value
        # - values[t + 1]: Critic's prediction of future value
        # - gamma: Discount factor (how much we value future vs immediate rewards)
        # - (1 - dones[t]): Zero out future value if episode ended
        # For CSTR: Expected future rewards from reactor state after temperature adjustment
        
        # values[t]: Critic's prediction of current state value
        # For CSTR: Expected reward from current reactor conditions
        
        # The complete delta calculation:
        # "How much better/worse was the actual outcome compared to expectations?"
        delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
        
        # ===== COMPUTE GAE (GENERALIZED ADVANTAGE ESTIMATION) =====
        # GAE formula: A_t = δ_t + γλ(1 - done_t)A_{t+1}
        # This recursively combines immediate advantage with future advantages
        
        # delta: Immediate advantage (how much better/worse than expected)
        # gamma * lam * (1 - dones[t]) * last_gae: Discounted future advantage
        # - lam: GAE parameter (balances bias vs variance)
        # - (1 - dones[t]): Zero out future advantage if episode ended
        # - last_gae: Advantage from next timestep (computed in previous iteration)
        
        # For CSTR: Combines immediate temperature adjustment performance with
        # expected future performance from subsequent adjustments
        
        # ===== CHAINED ASSIGNMENT EXPLANATION =====
        # This line uses "chained assignment" - a valid Python feature
        # advantages[t] = last_gae = delta + gamma * lam * (1 - dones[t]) * last_gae
        #
        # **How Chained Assignment Works:**
        # 1. Calculate the rightmost expression: delta + gamma * lam * (1 - dones[t]) * last_gae
        # 2. Assign the result to last_gae (for next iteration)
        # 3. Assign the same result to advantages[t] (for current timestep)
        #
        # **Benefits of Chained Assignment:**
        # - Concise: One line instead of three
        # - Clear intent: Shows both variables should have the same value
        # - Efficient: Calculation happens once, result used twice
        # - Common pattern: Widely used in Python for this purpose
        advantages[t] = last_gae = delta + gamma * lam * (1 - dones[t]) * last_gae
    
    # ===== COMPUTE RETURNS =====
    # Returns = Advantages + Values
    # Returns represent the total expected future rewards from each state
    # For CSTR: Total expected future conversion efficiency from each reactor state
    returns = advantages + values[:-1]  # values[:-1] excludes the bootstrap zero
    
    # ===== ADVANTAGE NORMALIZATION (CRITICAL FOR STABILITY) =====
    # Normalize advantages to mean=0, std=1 for stable training
    # This prevents advantages from becoming too large or small
    # Critical for PPO stability - unnormalized advantages can cause training instability
    
    # advantages.mean(): Average advantage across all timesteps
    # advantages.std(ddof=1): Sample standard deviation (consistent with torch.std())
    # 1e-8: Small constant to prevent division by zero
    # For CSTR: Normalizes temperature adjustment performance relative to average performance
    # Note: Using ddof=1 to match torch.std() behavior (sample std, not population std)
    advantages_normalized = (advantages - advantages.mean()) / (advantages.std(ddof=1) + 1e-8)
    
    # ===== RETURN RESULTS =====
    # Convert to PyTorch tensors for neural network training
    # torch.FloatTensor(): Converts numpy arrays to PyTorch tensors
    # For CSTR: Returns normalized advantages and total returns for policy training
    return torch.FloatTensor(advantages_normalized), torch.FloatTensor(returns), torch.FloatTensor(advantages)


# ===== STEP 4 & 5: PPO UPDATE (CLIPPED OBJECTIVE & VALUE LOSS) =====
# This is the core learning mechanism of PPO - improving the policy and value function
# using the collected experience and computed advantages
#
# **What is PPO Update?**
# PPO update takes the collected experience and uses it to improve both:
# 1. **Policy (Actor)**: How to choose actions based on states
# 2. **Value Function (Critic)**: How to estimate state values
#
# **Analogy: Fine-Tuning a Master Chef**
# - **Before**: Chef has certain cooking techniques (policy)
# - **During**: Chef tries new techniques based on feedback (advantages)
# - **Clipping**: Chef doesn't change too drastically (stays within 20% of original)
# - **Multiple Epochs**: Chef practices same recipes multiple times
# - **After**: Chef has refined techniques based on what worked well
#
# **Key Innovation: Clipped Surrogate Objective**
# PPO's main innovation is preventing the policy from changing too aggressively,
# which stabilizes training and prevents performance collapse.

def ppo_update(
    model: ActorCriticNet,
    states: list[np.ndarray],
    actions: list[np.ndarray],
    log_probs_old: list[float],
    returns: torch.Tensor,
    advantages: torch.Tensor,
    clip: float = 0.2,
    epochs: int = 10
) -> None:
    """
    Perform PPO policy and value function updates using collected experience.
    
    **What is PPO Update?**
    This function implements the core learning mechanism of PPO, updating both:
    1. **Actor (Policy)**: How to choose actions based on states
    2. **Critic (Value Function)**: How to estimate state values
    
    **For CSTR Context:**
    - **Actor Update**: Improves how to choose cooling temperature adjustments
    - **Critic Update**: Improves estimation of reactor state values
    - **Clipping**: Prevents drastic changes to temperature control strategy
    
    **Key Innovation: Clipped Surrogate Objective**
    PPO's main innovation is the clipped surrogate objective, which prevents
    the policy from changing too aggressively. This stabilizes training and
    prevents the performance collapse that can occur with other policy gradient methods.
    
    **Mathematical Formula:**
    L^CLIP(θ) = E[min(r_t(θ)A_t, clip(r_t(θ), 1-ε, 1+ε)A_t)]
    where r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)
    
    **Why Multiple Epochs?**
    - More efficient use of collected experience
    - Allows policy to learn from the same data multiple times
    - Improves sample efficiency compared to single-pass methods
    
    **Why Clipping?**
    - Prevents policy from changing too drastically
    - Maintains training stability
    - Allows for more aggressive learning rates
    
    Args:
        model (ActorCriticNet): Current policy and value function
        states (list): List of states from collected experience
        actions (list): List of actions taken in those states
        log_probs_old (list): Log probabilities of actions under old policy
        returns (torch.Tensor): Computed returns for each state
        advantages (torch.Tensor): Computed advantages for each action
        clip (float): Clipping parameter (default: 0.2 = 20% max change)
        epochs (int): Number of update epochs (default: 10)
    
    **Example:**
        >>> ppo_update(model, states, actions, log_probs_old, returns, advantages)
        >>> # Updates policy to choose better temperature adjustments
        >>> # Updates critic to better estimate reactor state values
    """
    
    # ===== DATA PREPARATION =====
    # Convert input data to PyTorch tensors for neural network processing
    # torch.FloatTensor(): Converts lists/arrays to PyTorch tensors
    # Required because neural networks expect tensor inputs
    states = torch.FloatTensor(states)
    actions = torch.FloatTensor(actions)
    log_probs_old = torch.FloatTensor(log_probs_old)
    
    # ===== MULTIPLE EPOCHS OF POLICY IMPROVEMENT =====
    # Run multiple epochs to make efficient use of collected experience
    # Each epoch: forward pass → compute losses → update networks
    # For CSTR: Practice the same temperature control decisions multiple times
    for _ in range(epochs):
        
        # ===== FORWARD PASS THROUGH ACTOR-CRITIC NETWORK =====
        # Get current policy and value predictions for all states
        # model(states): Returns (action_means, action_stds, state_values)
        # - action_means: Current policy's suggested actions
        # - action_stds: Current policy's exploration parameters
        # - state_values: Current critic's value estimates
        # For CSTR: Predicts optimal temperature adjustments and reactor values
        mean, std, values = model(states)
        
        # ===== CREATE ACTION DISTRIBUTION =====
        # Create probability distribution for action evaluation
        # torch.distributions.Normal(mean, std): Creates Gaussian distribution
        # This represents the current policy's action selection
        # For CSTR: Distribution over cooling temperature adjustments
        dist = torch.distributions.Normal(mean, std)
        
        # ===== COMPUTE NEW ACTION PROBABILITIES =====
        # Calculate log probabilities of the taken actions under current policy
        # dist.log_prob(actions): Log probability of each action under current policy
        # .sum(dim=-1): Sum across action dimensions (if multi-dimensional)
        # For CSTR: Log probability of each temperature adjustment under current policy
        log_probs_new = dist.log_prob(actions).sum(dim=-1)
        
        # ===== COMPUTE ENTROPY FOR EXPLORATION =====
        # Calculate entropy of the action distribution
        # dist.entropy(): Measures uncertainty/exploration in the policy
        # .mean(): Average entropy across all samples
        # For CSTR: How much the policy explores different temperature adjustments
        entropy = dist.entropy().mean()
        
        # ===== COMPUTE IMPORTANCE SAMPLING RATIO =====
        # Calculate ratio: π_new(a|s) / π_old(a|s)
        # This measures how much the policy has changed for each action
        # torch.exp(log_probs_new - log_probs_old): exp(log_new - log_old) = new/old
        # For CSTR: How much the temperature control strategy has changed
        ratios = torch.exp(log_probs_new - log_probs_old)
        
        # ===== CLIPPED SURROGATE OBJECTIVE (PPO'S KEY INNOVATION) =====
        # This is the core of PPO - preventing policy from changing too drastically
        
        # surrogate1: Standard policy gradient objective
        # ratios * advantages: Standard importance sampling
        # For CSTR: Standard improvement of temperature control based on performance
        surrogate1 = ratios * advantages
        
        # surrogate2: Clipped policy gradient objective
        # torch.clamp(ratios, 1-clip, 1+clip): Limits ratio to [0.8, 1.2] for clip=0.2
        # This prevents the policy from changing too drastically
        # For CSTR: Limited improvement to prevent drastic changes in temperature control
        surrogate2 = torch.clamp(ratios, 1 - clip, 1 + clip) * advantages
        
        # actor_loss: Take the minimum of clipped and unclipped objectives
        # -torch.min(surrogate1, surrogate2).mean(): Negative because we maximize
        # This ensures we don't make changes that are too large
        # For CSTR: Conservative improvement of temperature control strategy
        actor_loss = -torch.min(surrogate1, surrogate2).mean()
        
        # ===== CRITIC LOSS (VALUE FUNCTION IMPROVEMENT) =====
        # Improve the critic's ability to estimate state values
        # F.mse_loss(values.squeeze(), returns): Mean squared error loss
        # - values.squeeze(): Critic's predictions (remove extra dimensions)
        # - returns: Actual returns computed from experience
        # For CSTR: Improve estimation of reactor state values
        critic_loss = F.mse_loss(values.squeeze(), returns)
        
        # ===== TOTAL LOSS WITH ENTROPY BONUS =====
        # Combine actor loss, critic loss, and entropy bonus
        # actor_loss: Policy improvement (main objective)
        # 0.5 * critic_loss: Value function improvement (weighted)
        # -0.01 * entropy: Encourage exploration (small penalty)
        # For CSTR: Balance between improving temperature control and maintaining exploration
        total_loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
        
        # ===== IMPORTANT: SEPARATE OPTIMIZATION STRATEGY =====
        # NOTE: We compute the total_loss but DON'T use it for optimization!
        # Instead, we optimize actor and critic separately for better control.
        #
        # **Why Separate Optimization?**
        # 1. **Different Learning Rates**: Actor and critic often need different learning rates
        # 2. **Different Objectives**: Actor learns policy, critic learns value function
        # 3. **Stability**: Prevents one network from interfering with the other
        # 4. **Control**: Can apply different regularization to each network
        #
        # **What Actually Happens:**
        # - actor_loss.backward(): Computes gradients for actor parameters only
        # - critic_loss.backward(): Computes gradients for critic parameters only
        # - actor_optimizer.step(): Updates only actor parameters
        # - critic_optimizer.step(): Updates only critic parameters
        #
        # **Why Compute total_loss Then?**
        # - For monitoring/logging purposes
        # - To track overall training progress
        # - For potential future use (some implementations do use it)
        # - For debugging and analysis
        #
        # **Alternative Approach (Not Used Here):**
        # If we wanted to use total_loss, we would do:
        # total_loss.backward()
        # actor_optimizer.step()  # Would update both actor and critic
        # But this gives less control and can be less stable
        
        # ===== ACTOR UPDATE (POLICY IMPROVEMENT) =====
        # Update the policy network to choose better actions
        # We use ONLY the actor_loss for this update, not the total_loss
        
        # Clear previous gradients for actor parameters
        # actor_optimizer.zero_grad(): Reset gradients to zero
        # Required before computing new gradients
        actor_optimizer.zero_grad()
        
        # Compute gradients for actor loss ONLY
        # actor_loss.backward(retain_graph=True): Compute gradients for actor parameters
        # retain_graph=True: Keep computational graph for critic update
        # For CSTR: Compute gradients for improving temperature control strategy
        actor_loss.backward(retain_graph=True)
        
        # Gradient clipping for stability
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5): Clip gradient norm
        # Prevents exploding gradients that could destabilize training
        # For CSTR: Prevent drastic changes to temperature control parameters
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        
        # Apply gradient updates to actor parameters ONLY
        # actor_optimizer.step(): Update actor network weights
        # This improves the policy based on the computed gradients
        # For CSTR: Update temperature control strategy based on performance
        actor_optimizer.step()
        
        # ===== CRITIC UPDATE (VALUE FUNCTION IMPROVEMENT) =====
        # Update the value function network to better estimate state values
        # We use ONLY the critic_loss for this update, not the total_loss
        
        # Clear previous gradients for critic parameters
        # critic_optimizer.zero_grad(): Reset gradients to zero
        # Required before computing new gradients
        critic_optimizer.zero_grad()
        
        # Compute gradients for critic loss ONLY
        # critic_loss.backward(): Compute gradients for critic parameters
        # No retain_graph needed since this is the final backward pass
        # For CSTR: Compute gradients for improving reactor state value estimation
        critic_loss.backward()
        
        # Apply gradient updates to critic parameters ONLY
        # critic_optimizer.step(): Update critic network weights
        # This improves the value function based on the computed gradients
        # For CSTR: Update reactor state value estimation based on actual performance
        critic_optimizer.step()


# # ===== MAIN TRAINING LOOP: COMPLETE PPO ALGORITHM =====
# # This is where all the PPO components come together to create a complete
# # reinforcement learning system for CSTR optimization
# #
# # **PPO Algorithm Overview:**
# # PPO follows a clear iterative loop that balances exploration, learning, and stability:
# # 1. **Collect Experience** (Rollouts) → 2. **Compute Advantages** (GAE) → 3. **Update Policy** (Clipped Objective)
# #
# # **For CSTR Context:**
# # - **Rollouts**: Test current temperature control strategy in reactor
# # - **Advantages**: Evaluate how well each temperature adjustment performed
# # - **Updates**: Improve temperature control strategy based on performance
# #
# # **Key PPO Principles Implemented:**
# # - **Proximal Policy Optimization**: Prevents drastic policy changes
# # - **Actor-Critic Architecture**: Separate policy and value learning
# # - **Generalized Advantage Estimation**: Stable advantage computation
# # - **Multiple Epochs**: Efficient use of collected experience
# # - **Separate Optimizers**: Independent control of policy and value learning

# # ===== TRAINING CONFIGURATION =====
# # num_updates: Total number of PPO update cycles
# # Each update: collect data → compute advantages → update policy
# # For CSTR: 1000 updates = 1000 cycles of temperature control improvement
# num_updates = 1000

# # ===== MAIN PPO TRAINING LOOP =====
# # This loop implements the complete PPO algorithm
# # Each iteration represents one complete cycle of the PPO algorithm
# for update in range(num_updates):
    
#     # ===== STEP 1: EXPERIENCE COLLECTION (ROLLOUTS) =====
#     # Collect trajectories using the current policy
#     # This is the "data collection" phase of PPO
#     # For CSTR: Test current temperature control strategy in the reactor
#     # Returns: states, actions, rewards, dones, values, log_probs_old
#     # - states: Reactor conditions [Ca, Cb, T] at each timestep
#     # - actions: Temperature adjustments applied at each timestep
#     # - rewards: Conversion efficiency and safety rewards received
#     # - dones: Whether reactor reached unsafe conditions or time limit
#     # - values: Critic's predictions of expected future rewards
#     # - log_probs_old: Action probabilities under the current policy
#     states, actions, rewards, dones, values, log_probs_old = collect_trajectories(model, env)
    
#     # ===== STEP 2: ADVANTAGE COMPUTATION (GAE) =====
#     # Compute advantages using Generalized Advantage Estimation
#     # This is the "learning signal" phase of PPO
#     # For CSTR: Evaluate how much better/worse each temperature adjustment was than expected
#     # Returns: advantages, returns
#     # - advantages: How much better/worse actions were than expected (normalized)
#     # - returns: Total expected future rewards from each state
#     advantages, returns = compute_gae(rewards, dones, values)
    
#     # ===== STEP 3: POLICY AND VALUE FUNCTION UPDATE =====
#     # Update both the policy (actor) and value function (critic)
#     # This is the "learning" phase of PPO
#     # For CSTR: Improve temperature control strategy and reactor state estimation
#     # - model: Current actor-critic network
#     # - states: Reactor conditions from collected experience
#     # - actions: Temperature adjustments from collected experience
#     # - log_probs_old: Action probabilities under old policy (for importance sampling)
#     # - returns: Total future rewards (for critic learning)
#     # - advantages: How much better/worse actions were (for actor learning)
#     ppo_update(model, states, actions, log_probs_old, returns, advantages)
    
#     # ===== PROGRESS MONITORING =====
#     # Print progress every 100 updates
#     # This helps track training progress and identify potential issues
#     # For CSTR: Monitor temperature control strategy improvement over time
#     if (update + 1) % 100 == 0:
#         print(f"Update {update + 1}/{num_updates} completed.")

# # ===== MODEL PERSISTENCE =====
# # Save the trained actor-critic model after training
# # This preserves the learned policy and value function for later use
# # For CSTR: Save the optimized temperature control strategy
# # torch.save(model.state_dict(), "ppo_actor_critic_cstr.pth"):
# # - model.state_dict(): Extracts all network parameters (weights and biases)
# # - "ppo_actor_critic_cstr.pth": File path to save the model
# # - Both actor and critic parameters are saved together
# torch.save(model.state_dict(), "ppo_actor_critic_cstr.pth")  # clearly storing both actor and critic parameters
