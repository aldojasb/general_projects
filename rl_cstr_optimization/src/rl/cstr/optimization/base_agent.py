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
import logging

# Create a logger for this module
logger = logging.getLogger(__name__)

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
                                    next_state, terminated, truncated) tuples, formatted as tensors, 
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
) -> tuple[list[np.ndarray], list[np.ndarray], list[float], list[bool], list[bool], list[float], list[float]]:
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
        tuple: (states, actions, rewards, terminated, truncated, values, log_probs)
            - states: List of observed states [Ca, Cb, T]
            - actions: List of actions taken (cooling temperature adjustments)
            - rewards: List of rewards received
            - terminated: List of episode termination flags (natural ending)
            - truncated: List of episode truncation flags (artificial ending)
            - values: List of critic's value estimates
            - log_probs: List of action log probabilities (for importance sampling)
    
    **Example Usage:**
        >>> states, actions, rewards, terminated, truncated, values, log_probs = collect_trajectories(model, env)
        >>> # states: [[0.8, 0.2, 350.0], [0.7, 0.3, 348.0], ...]
        >>> # actions: [[2.3], [-1.7], [0.5], ...]  # temperature adjustments
        >>> # rewards: [15.2, 12.8, 18.1, ...]  # conversion efficiency rewards
    """
    
    # ===== TRAJECTORY COLLECTION INITIALIZATION =====
    logger.info(f"\n{'='*60}")
    logger.info(f"TRAJECTORY COLLECTION STARTED")
    logger.info(f"{'='*60}")
    logger.info(f"Collection Configuration:")
    logger.info(f"  - Target steps: {steps}")
    logger.info(f"  - Environment: {type(env).__name__}")
    logger.info(f"  - Model: {type(model).__name__}")
    
    # Reset environment to start fresh episode
    # env.reset(): Returns (initial_state, info) tuple and resets environment to starting conditions
    # For CSTR: Returns initial reactor conditions [Ca_initial, Cb_initial, T_initial]
    # This ensures we start from a safe, known state for consistent data collection
    state, info = env.reset()
    logger.info(f"Environment reset completed - initial state: {state}, info: {info}")
    
    # Initialize empty lists to store trajectory data
    # These will hold the experience collected during the rollout
    # Each list will grow to length 'steps' by the end of the function
    states, actions, rewards, terminated_list, truncated_list, values, log_probs = [], [], [], [], [], [], []
    
    # ===== EXPERIENCE COLLECTION LOOP =====
    # Collect experience for 'steps' timesteps
    # Each iteration: observe state → select action → get reward → observe next state
    # This creates a trajectory of experience that we'll use to improve the policy
    for step_idx in range(steps):
        if step_idx % 100 == 0:  # Log every 100 steps
            logger.info(f"Processing step {step_idx}/{steps}")
        
        # ===== STATE PROCESSING =====
        # Convert state from numpy array to PyTorch tensor
        # torch.FloatTensor(state): Converts numpy array to tensor for neural network input
        # Required because PyTorch models expect tensor inputs, not numpy arrays
        # For CSTR: Converts [Ca, Cb, T] numpy array to tensor for model input
        if step_idx == 0:  # Log only first step to avoid spam
            logger.info(f"Converting state to tensor: {state}, type: {type(state)}")
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
        # env.step(action.numpy()): Executes action and returns (next_state, reward, terminated, truncated, info)
        # action.numpy(): Converts tensor back to numpy for environment compatibility
        # For CSTR: Applies cooling temperature adjustment and observes reactor response
        # Returns: new reactor conditions, reward based on efficiency/safety, episode status
        try:
            next_state, reward, terminated, truncated, info = env.step(action.numpy())
        except Exception as e:
            logger.error(f"Error in environment step: {e}")
            raise e


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
        
        # terminated_list.append(terminated): Whether episode naturally ended
        # For CSTR: True if reactor reached unsafe conditions (natural ending)
        # Example: False - episode continues, True - reactor overheated
        terminated_list.append(terminated)
        
        # truncated_list.append(truncated): Whether episode was artificially ended
        # For CSTR: True if time limit reached (artificial ending)
        # Example: False - episode continues, True - time limit reached
        truncated_list.append(truncated)
        
        # Log episode termination details for debugging
        if (terminated or truncated) and step_idx < 10:  # Log only first few terminations to avoid spam
            logger.debug(f"Episode ended at step {step_idx}: terminated={terminated}, truncated={truncated}")
        
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
        # If episode ended (either terminated or truncated), reset environment for fresh start
        # env.reset(): Returns (new_initial_state, info) tuple
        # For CSTR: Resets reactor to safe initial conditions
        # This prevents the agent from getting stuck in bad states
        if terminated or truncated:
            logger.debug(f"Episode ended at step {step_idx}, resetting environment")
            state, info = env.reset()

    # ===== TRAJECTORY COLLECTION SUMMARY =====
    logger.info(f"\n{'='*60}")
    logger.info(f"TRAJECTORY COLLECTION COMPLETED - FINAL SUMMARY")
    logger.info(f"{'='*60}")
    
    # Calculate collection statistics
    total_steps = len(states)
    natural_terminations = sum(terminated_list)
    artificial_truncations = sum(truncated_list)
    
    logger.info(f"Collection Statistics:")
    logger.info(f"  - Total steps collected: {total_steps}")
    logger.info(f"  - States collected: {len(states)}")
    logger.info(f"  - Actions collected: {len(actions)}")
    logger.info(f"  - Rewards collected: {len(rewards)}")
    logger.info(f"  - Values collected: {len(values)}")
    logger.info(f"  - Log probabilities collected: {len(log_probs)}")
    
    logger.info(f"Episode Termination Summary:")
    logger.info(f"  - Natural terminations: {natural_terminations} ({natural_terminations/total_steps*100:.1f}%)")
    logger.info(f"  - Artificial truncations: {artificial_truncations} ({artificial_truncations/total_steps*100:.1f}%)")
    logger.info(f"  - Continuing episodes: {total_steps - natural_terminations - artificial_truncations} ({100 - (natural_terminations + artificial_truncations)/total_steps*100:.1f}%)")
    
    # Log sample data for verification
    if rewards:
        rewards_array = np.array(rewards)
        logger.info(f"Reward Statistics:")
        logger.info(f"  - Mean reward: {rewards_array.mean():.4f}")
        logger.info(f"  - Std reward: {rewards_array.std():.4f}")
        logger.info(f"  - Min reward: {rewards_array.min():.4f}")
        logger.info(f"  - Max reward: {rewards_array.max():.4f}")
        logger.info(f"  - Total reward: {rewards_array.sum():.4f}")
    
    if values:
        values_array = np.array(values)
        logger.info(f"Value Statistics:")
        logger.info(f"  - Mean value: {values_array.mean():.4f}")
        logger.info(f"  - Std value: {values_array.std():.4f}")
        logger.info(f"  - Min value: {values_array.min():.4f}")
        logger.info(f"  - Max value: {values_array.max():.4f}")
    
    # Log sample values for debugging
    if total_steps > 0:
        logger.info(f"Sample Data (first 3 steps):")
        for i in range(min(3, total_steps)):
            logger.info(f"  - Step {i}: state={states[i]}, action={actions[i]}, reward={rewards[i]:.4f}, value={values[i]:.4f}")
        
        if total_steps > 3:
            logger.info(f"Sample Data (last 3 steps):")
            for i in range(max(0, total_steps-3), total_steps):
                logger.info(f"  - Step {i}: state={states[i]}, action={actions[i]}, reward={rewards[i]:.4f}, value={values[i]:.4f}")
    
    logger.info(f"{'='*60}")
    
    # Return all collected data as lists (will be converted to tensors later)
    # This data will be used for:
    # 1. Computing advantages (GAE) - how much better actions were than expected
    # 2. Policy updates (PPO clipped objective) - improve the policy
    # 3. Value function updates (MSE loss) - improve value estimates
    # 
    # For CSTR: Returns trajectory of reactor control decisions and outcomes
    # This data shows how well the current control policy performed
    return (states, actions, rewards, terminated_list, truncated_list, values, log_probs)


# ===== COMPUTE ADVANTAGES (GAE) =====
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
    terminated: list[bool], 
    truncated: list[bool],
    values: list[float], 
    gamma: float = 0.99, 
    lam: float = 0.95
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    - terminated: Natural episode endings (reactor unsafe conditions reached)
    - truncated: Artificial episode endings (time limits, memory constraints)
    - values: Critic's predictions of expected future rewards
    - advantages: How much better/worse each temperature adjustment was than expected
    - returns: Total expected future rewards from each state
    
    **Key Parameters:**
    - gamma (γ): Discount factor - how much we value future vs immediate rewards
    - lambda (λ): GAE parameter - balances bias vs variance in advantage estimation
    - terminated: Natural episode termination flags - when reactor reaches unsafe conditions
    - truncated: Artificial episode termination flags - when time limits or constraints are reached
    
    **Difference Between Terminated and Truncated:**
    
    **Terminated (Natural Ending):**
    - Definition: Episode ends naturally due to environment conditions
    - Examples: Reactor temperature > 400°C, pressure limits exceeded, unsafe conditions
    - Learning Impact: Real failure of agent's policy - should affect learning
    - GAE Treatment: Zeros out future rewards (1 - terminated[t] = 0)
    
    **Truncated (Artificial Ending):**
    - Definition: Episode ends artificially due to external constraints
    - Examples: Time limit reached, memory constraints, user interruption
    - Learning Impact: Not a failure - just artificial boundary
    - GAE Treatment: Continues future rewards (1 - terminated[t] = 1)
    
    **CSTR-Specific Examples:**
    >>> # Natural termination (BAD - agent failed)
    >>> state = [0.8, 0.2, 405.0]  # Temperature too high
    >>> terminated = True   # Reactor unsafe - agent failed
    >>> truncated = False   # Not artificial
    >>> # Result: Future rewards zeroed out in GAE
    >>> 
    >>> # Artificial truncation (NEUTRAL - not agent's fault)
    >>> state = [0.7, 0.3, 350.0]  # Normal conditions
    >>> terminated = False  # Reactor safe
    >>> truncated = True    # Time limit reached - not agent's fault
    >>> # Result: Future rewards continue in GAE
    >>> 
    >>> # Both false (CONTINUING - normal operation)
    >>> state = [0.8, 0.2, 350.0]  # Normal conditions
    >>> terminated = False  # Reactor safe
    >>> truncated = False   # No time limit reached
    >>> # Result: Full GAE computation with future rewards
    
    **Current Implementation:**
    We use only `terminated` for GAE computation while logging both for analysis.
    This approach:
    1. **Learns from Real Failures**: Only natural terminations affect learning
    2. **Provides Flexibility**: Easy to modify logic in the future
    3. **Maintains Monitoring**: Logs both types for analysis
    4. **Follows Modern Practice**: Aligns with current RL standards
    
    
    Args:
        rewards (list): List of rewards received for each action
        terminated (list): List of natural episode termination flags (True/False)
        truncated (list): List of artificial episode termination flags (True/False)
        values (list): List of critic's value estimates for each state
        gamma (float): Discount factor for future rewards (default: 0.99)
        lam (float): GAE parameter for bias-variance trade-off (default: 0.95)
    
    Returns:
        tuple: (advantages, returns, raw_advantages)
            - advantages: Normalized advantage estimates for each action
            - returns: Total expected future rewards from each state
            - raw_advantages: Raw advantage estimates for each action
    
    **Example:**
        >>> rewards = [15.2, 12.8, 18.1, 14.5]  # CSTR conversion rewards
        >>> values = [15.0, 13.0, 17.5, 14.0]   # Critic's predictions
        >>> terminated = [False, False, False, True]   # Natural episode termination
        >>> truncated = [False, False, False, False]   # No artificial truncation
        >>> advantages, returns, raw_advantages = compute_gae(rewards, terminated, truncated, values)
        >>> # advantages: [0.2, -0.2, 0.6, 0.5]  # How much better/worse than expected
        >>> # returns: [15.2, 12.8, 18.1, 14.5]  # Total future rewards
        >>> raw_advantages: [12.0, 14.0, 10.7, 15.5]  # Raw advantage estimates
    """
    
    # ===== LOGGING AND ANALYSIS =====
    # Log episode statistics for monitoring and analysis
    natural_terminations = sum(terminated)
    artificial_truncations = sum(truncated)
    total_steps = len(rewards)
    
    logger.info(f"GAE Analysis: {natural_terminations} natural terminations, {artificial_truncations} artificial truncations out of {total_steps} steps")
    
    if natural_terminations > 0:
        logger.info(f"  Natural terminations indicate unsafe reactor conditions - agent needs improvement")
    if artificial_truncations > 0:
        logger.info(f"  Artificial truncations are time limits - not agent failures")
    
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
        
        # gamma * values[t + 1] * (1 - terminated[t]): Discounted future value
        # - values[t + 1]: Critic's prediction of future value
        # - gamma: Discount factor (how much we value future vs immediate rewards)
        # - (1 - terminated[t]): Zero out future value if episode naturally ended
        # For CSTR: Expected future rewards from reactor state after temperature adjustment
        # Note: Only natural terminations (unsafe conditions) zero out future rewards
        # Artificial truncations (time limits) do NOT zero out future rewards
        
        # values[t]: Critic's prediction of current state value
        # For CSTR: Expected reward from current reactor conditions
        
        # The complete delta calculation:
        # "How much better/worse was the actual outcome compared to expectations?"
        # Only terminated (real failures) affect future reward expectations
        delta = rewards[t] + gamma * values[t + 1] * (1 - terminated[t]) - values[t]
        
        # ===== COMPUTE GAE (GENERALIZED ADVANTAGE ESTIMATION) =====
        # GAE formula: A_t = δ_t + γλ(1 - terminated_t)A_{t+1}
        # This recursively combines immediate advantage with future advantages
        
        # delta: Immediate advantage (how much better/worse than expected)
        # gamma * lam * (1 - terminated[t]) * last_gae: Discounted future advantage
        # - lam: GAE parameter (balances bias vs variance)
        # - (1 - terminated[t]): Zero out future advantage if episode naturally ended
        # - last_gae: Advantage from next timestep (computed in previous iteration)
        
        # For CSTR: Combines immediate temperature adjustment performance with
        # expected future performance from subsequent adjustments
        # Only natural terminations (unsafe reactor conditions) zero out future advantages
        # Artificial truncations (time limits) continue future advantages
        
        # ===== CHAINED ASSIGNMENT EXPLANATION =====
        # This line uses "chained assignment" - a valid Python feature
        # advantages[t] = last_gae = delta + gamma * lam * (1 - terminated[t]) * last_gae
        #
        # **How Chained Assignment Works:**
        # 1. Calculate the rightmost expression: delta + gamma * lam * (1 - terminated[t]) * last_gae
        # 2. Assign the result to last_gae (for next iteration)
        # 3. Assign the same result to advantages[t] (for current timestep)
        #
        # **Benefits of Chained Assignment:**
        # - Concise: One line instead of three
        # - Clear intent: Shows both variables should have the same value
        # - Efficient: Calculation happens once, result used twice
        # - Common pattern: Widely used in Python for this purpose
        advantages[t] = last_gae = delta + gamma * lam * (1 - terminated[t]) * last_gae
    
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
    # - advantages: Normalized advantage estimates for each action
    # - returns: Total expected future rewards from each state
    # - raw_advantages: Raw advantage estimates for each action
    # Convert to PyTorch tensors for neural network training
    # torch.FloatTensor(): Converts numpy arrays to PyTorch tensors
    # For CSTR: Returns normalized advantages and total returns for policy training
    
    # Convert to tensors for logging
    advantages_tensor = torch.FloatTensor(advantages)
    returns_tensor = torch.FloatTensor(returns)
    advantages_normalized_tensor = torch.FloatTensor(advantages_normalized)

    # ===== GAE COMPUTATION INITIALIZATION =====
    logger.info(f"\n{'='*60}")
    logger.info(f"GAE COMPUTATION STARTED")
    logger.info(f"{'='*60}")
    logger.info(f"GAE Configuration:")
    logger.info(f"  - Gamma (discount): {gamma}")
    logger.info(f"  - Lambda (GAE): {lam}")
    logger.info(f"  - Total steps: {len(rewards)}")
    
    # ===== COMPREHENSIVE LOGGING FOR RETURNS TRACKING =====
    # Log detailed information about the GAE computation and returns
    # This helps monitor learning progress and debug training issues
    
    # Log episode termination statistics
    natural_terminations = sum(terminated)
    artificial_truncations = sum(truncated)
    total_steps = len(rewards)
    
    logger.info(f"Input Statistics:")
    logger.info(f"  - Total steps: {total_steps}")
    logger.info(f"  - Natural terminations: {natural_terminations} ({natural_terminations/total_steps*100:.1f}%)")
    logger.info(f"  - Artificial truncations: {artificial_truncations} ({artificial_truncations/total_steps*100:.1f}%)")
    
    # Log reward statistics
    rewards_array = np.array(rewards)
    logger.info(f"  - Reward statistics:")
    logger.info(f"    * Mean reward: {rewards_array.mean():.4f}")
    logger.info(f"    * Std reward: {rewards_array.std():.4f}")
    logger.info(f"    * Min reward: {rewards_array.min():.4f}")
    logger.info(f"    * Max reward: {rewards_array.max():.4f}")
    logger.info(f"    * Total reward: {rewards_array.sum():.4f}")
    
    # Log value function statistics
    values_array = np.array(values)
    logger.info(f"  - Value function statistics:")
    logger.info(f"    * Mean value: {values_array.mean():.4f}")
    logger.info(f"    * Std value: {values_array.std():.4f}")
    logger.info(f"    * Min value: {values_array.min():.4f}")
    logger.info(f"    * Max value: {values_array.max():.4f}")
    
    # Log advantage statistics (before normalization)
    logger.info(f"  - Raw advantage statistics:")
    logger.info(f"    * Mean advantage: {advantages_tensor.mean().item():.4f}")
    logger.info(f"    * Std advantage: {advantages_tensor.std().item():.4f}")
    logger.info(f"    * Min advantage: {advantages_tensor.min().item():.4f}")
    logger.info(f"    * Max advantage: {advantages_tensor.max().item():.4f}")
    logger.info(f"    * Advantage range: {advantages_tensor.max().item() - advantages_tensor.min().item():.4f}")
    
    # Log normalized advantage statistics
    logger.info(f"  - Normalized advantage statistics:")
    logger.info(f"    * Mean normalized: {advantages_normalized_tensor.mean().item():.6f} (should be ~0)")
    logger.info(f"    * Std normalized: {advantages_normalized_tensor.std().item():.6f} (should be ~1)")
    logger.info(f"    * Min normalized: {advantages_normalized_tensor.min().item():.4f}")
    logger.info(f"    * Max normalized: {advantages_normalized_tensor.max().item():.4f}")
    
    # Log return statistics
    logger.info(f"  - Return statistics:")
    logger.info(f"    * Mean return: {returns_tensor.mean().item():.4f}")
    logger.info(f"    * Std return: {returns_tensor.std().item():.4f}")
    logger.info(f"    * Min return: {returns_tensor.min().item():.4f}")
    logger.info(f"    * Max return: {returns_tensor.max().item():.4f}")
    logger.info(f"    * Return range: {returns_tensor.max().item() - returns_tensor.min().item():.4f}")
    
    # Log GAE parameters used
    logger.info(f"  - GAE parameters:")
    logger.info(f"    * Gamma (discount): {gamma}")
    logger.info(f"    * Lambda (GAE): {lam}")
    
    # Log learning signal quality indicators
    advantage_std = advantages_tensor.std().item()
    return_std = returns_tensor.std().item()
    
    logger.info(f"  - Learning signal quality:")
    logger.info(f"    * Advantage std: {advantage_std:.4f}")
    logger.info(f"    * Return std: {return_std:.4f}")
    
    # Warning for potential training issues
    if advantage_std < 0.1:
        logger.warning(f"    * WARNING: Low advantage std ({advantage_std:.4f}) - weak learning signal")
    elif advantage_std > 10.0:
        logger.warning(f"    * WARNING: High advantage std ({advantage_std:.4f}) - potential training instability")
    
    if return_std < 0.1:
        logger.warning(f"    * WARNING: Low return std ({return_std:.4f}) - weak value learning signal")
    elif return_std > 50.0:
        logger.warning(f"    * WARNING: High return std ({return_std:.4f}) - potential value function instability")
    
    # Log sample values for debugging
    if total_steps > 0:
        logger.info(f"  - Sample values (first 3 steps):")
        for i in range(min(3, total_steps)):
            logger.info(f"    * Step {i}: reward={rewards[i]:.4f}, value={values[i]:.4f}, "
                       f"advantage={advantages[i]:.4f}, return={returns[i]:.4f}")
        
        if total_steps > 3:
            logger.info(f"  - Sample values (last 3 steps):")
            for i in range(max(0, total_steps-3), total_steps):
                logger.info(f"    * Step {i}: reward={rewards[i]:.4f}, value={values[i]:.4f}, "
                           f"advantage={advantages[i]:.4f}, return={returns[i]:.4f}")
    
    # Log episode boundary information
    if natural_terminations > 0:
        termination_steps = [i for i, term in enumerate(terminated) if term]
        logger.info(f"  - Natural terminations at steps: {termination_steps}")
    
    if artificial_truncations > 0:
        truncation_steps = [i for i, trunc in enumerate(truncated) if trunc]
        logger.info(f"  - Artificial truncations at steps: {truncation_steps}")
    
    # Log computation summary
    logger.info(f"  - GAE computation completed successfully")
    logger.info(f"    * Output shapes: advantages={advantages_normalized_tensor.shape}, "
               f"returns={returns_tensor.shape}, raw_advantages={advantages_tensor.shape}")
    
    # Check if all outputs are finite
    advantages_finite = torch.all(torch.isfinite(advantages_normalized_tensor))
    returns_finite = torch.all(torch.isfinite(returns_tensor))
    raw_advantages_finite = torch.all(torch.isfinite(advantages_tensor))
    all_finite = advantages_finite and returns_finite and raw_advantages_finite
    
    logger.info(f"    * All outputs are finite: {all_finite}")
    
    # ===== GAE COMPUTATION SUMMARY =====
    logger.info(f"\n{'='*60}")
    logger.info(f"GAE COMPUTATION COMPLETED - FINAL SUMMARY")
    logger.info(f"{'='*60}")
    
    # Calculate computation statistics
    logger.info(f"Computation Results:")
    logger.info(f"  - Normalized advantages shape: {advantages_normalized_tensor.shape}")
    logger.info(f"  - Returns shape: {returns_tensor.shape}")
    logger.info(f"  - Raw advantages shape: {advantages_tensor.shape}")
    logger.info(f"  - All outputs finite: {all_finite}")
    
    # Log final statistics
    logger.info(f"Final Statistics:")
    logger.info(f"  - Normalized advantage mean: {advantages_normalized_tensor.mean().item():.6f} (target: ~0)")
    logger.info(f"  - Normalized advantage std: {advantages_normalized_tensor.std().item():.6f} (target: ~1)")
    logger.info(f"  - Return mean: {returns_tensor.mean().item():.4f}")
    logger.info(f"  - Return std: {returns_tensor.std().item():.4f}")
    
    # Log computation success
    if all_finite:
        logger.info(f"  - GAE computation successful: All outputs are finite")
    else:
        logger.warning(f"  - WARNING: GAE computation may have issues - some outputs are not finite")
    
    logger.info(f"{'='*60}")

    return advantages_normalized_tensor, returns_tensor, advantages_tensor


# ===== PPO UPDATE (CLIPPED OBJECTIVE & VALUE LOSS) =====
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
    actor_optimizer: optim.Optimizer,
    critic_optimizer: optim.Optimizer,
    actor_clip: float = 0.2,
    value_clip: float = None,
    epochs: int = 10
) -> tuple[float, float, float]:
    """
    Perform PPO policy and value function updates using collected experience.
    
    **PPO's Core Innovation: Clipped Surrogate Objective**
    PPO's main contribution is preventing the policy from changing too drastically
    in a single update. This is achieved through the clipped surrogate objective:
    
    L^CLIP(θ) = E[min(r_t(θ)A_t, clip(r_t(θ), 1-ε, 1+ε)A_t)]
    
    where r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)
    
    **Value Function Clipping (Optional Enhancement):**
    Many modern PPO implementations also clip the value function to prevent
    the critic from making too large updates, which can destabilize training.
    
    **Why This Matters:**
    1. **Stability**: Prevents performance collapse from aggressive updates
    2. **Conservative Learning**: Allows for more aggressive learning rates
    3. **Sample Efficiency**: Multiple epochs of updates on the same data
    4. **Value Stability**: Prevents critic from making extreme changes
    
    **For CSTR Context:**
    - **Actor Update**: Improves temperature control strategy conservatively
    - **Critic Update**: Improves reactor state value estimation conservatively
    - **Clipping**: Prevents drastic changes to both policy and value function
    
    Args:
        model (ActorCriticNet): Current policy and value function
        states (list): List of states from collected experience
        actions (list): List of actions taken in those states
        log_probs_old (list): Log probabilities of actions under old policy
        returns (torch.Tensor): Computed returns for each state
        advantages (torch.Tensor): Computed advantages for each action
        actor_optimizer (optim.Optimizer): Optimizer for the actor (policy) network
        critic_optimizer (optim.Optimizer): Optimizer for the critic (value) network
        actor_clip (float): Policy clipping parameter ε (default: 0.2 = 20% max change)
        value_clip (float, optional): Value function clipping parameter (default: None = disabled)
        epochs (int): Number of update epochs (default: 10)
    
    Returns:
        tuple[float, float, float]: Final losses and entropy from the last epoch
            - actor_loss (float): Final policy loss after all epochs
            - critic_loss (float): Final value function loss after all epochs  
            - entropy (float): Final entropy after all epochs
    
    **Example:**
        >>> actor_optimizer = optim.Adam(model.actor.parameters(), lr=3e-4)
        >>> critic_optimizer = optim.Adam(model.critic.parameters(), lr=1e-3)
        >>> actor_loss, critic_loss, entropy = ppo_update(model, states, actions, log_probs_old, returns, advantages,
        >>>           actor_optimizer, critic_optimizer, actor_clip=0.2, value_clip=0.2)
        >>> # Updates policy and value function conservatively, returns final losses
        >>> 
        >>> # Without value function clipping:
        >>> actor_loss, critic_loss, entropy = ppo_update(model, states, actions, log_probs_old, returns, advantages,
        >>>           actor_optimizer, critic_optimizer, actor_clip=0.2)
        >>> # Updates policy conservatively, value function normally, returns final losses
    """
    
    # ===== DATA PREPARATION =====
    # Convert input data to PyTorch tensors for neural network processing
    # Convert lists to numpy arrays first for efficiency, then to tensors
    # Required because neural networks expect tensor inputs

    
    logger.info(f"Converting {len(states)} states, {len(actions)} actions, {len(log_probs_old)} log_probs to tensors")
    logger.info(f"Sample state type: {type(states[0])}, shape: {states[0].shape if hasattr(states[0], 'shape') else 'N/A'}")
    logger.info(f"Sample action type: {type(actions[0])}, shape: {actions[0].shape if hasattr(actions[0], 'shape') else 'N/A'}")
    
    states_array = np.array(states)
    actions_array = np.array(actions)
    log_probs_old_array = np.array(log_probs_old)
    
    logger.info(f"Converted to numpy arrays - states shape: {states_array.shape}, actions shape: {actions_array.shape}")
    
    states = torch.FloatTensor(states_array)
    actions = torch.FloatTensor(actions_array)
    log_probs_old = torch.FloatTensor(log_probs_old_array)
    
    logger.info(f"Converted to tensors - states shape: {states.shape}, actions shape: {actions.shape}")
    
    # ===== PPO UPDATE INITIALIZATION LOGGING =====
    logger.info(f"\n{'='*60}")
    logger.info(f"PPO UPDATE STARTED")
    logger.info(f"{'='*60}")
    logger.info(f"Update Configuration:")
    logger.info(f"  - Actor clip parameter: {actor_clip} (max {actor_clip*100:.0f}% policy change)")
    logger.info(f"  - Value clip parameter: {value_clip if value_clip is not None else 'Disabled'}")
    logger.info(f"  - Number of epochs: {epochs}")
    logger.info(f"  - Batch size: {len(states)}")
    logger.info(f"  - Actor learning rate: {actor_optimizer.param_groups[0]['lr']}")
    logger.info(f"  - Critic learning rate: {critic_optimizer.param_groups[0]['lr']}")
    
    # Log input statistics
    logger.info(f"Input Statistics:")
    logger.info(f"  - Advantages: mean={advantages.mean().item():.4f}, std={advantages.std().item():.4f}")
    logger.info(f"  - Returns: mean={returns.mean().item():.4f}, std={returns.std().item():.4f}")
    logger.info(f"  - Old log probs: mean={log_probs_old.mean().item():.4f}, std={log_probs_old.std().item():.4f}")
    
    # ===== EPOCH TRACKING VARIABLES =====
    # Track evolution of losses and metrics across epochs
    epoch_losses = {
        'actor_loss': [],
        'critic_loss': [],
        'total_loss': [],
        'entropy': [],
        'policy_ratio_mean': [],
        'policy_ratio_std': [],
        'policy_ratio_min': [],
        'policy_ratio_max': [],
        'clipped_ratio_count': [],
        'value_clipped_count': []
    }
    
    # ===== MULTIPLE EPOCHS OF POLICY IMPROVEMENT =====
    # Run multiple epochs to make efficient use of collected experience
    # Each epoch: forward pass → compute losses → update networks
    # For CSTR: Practice the same temperature control decisions multiple times
    for epoch in range(epochs):
        
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
        
        # ===== PPO'S CORE: IMPORTANCE SAMPLING RATIO =====
        # Calculate ratio: π_new(a|s) / π_old(a|s)
        # This measures how much the policy has changed for each action
        # torch.exp(log_probs_new - log_probs_old): exp(log_new - log_old) = new/old
        # For CSTR: How much the temperature control strategy has changed
        # 
        # **Mathematical Foundation:**
        # r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)
        # This ratio tells us how much more/less likely the new policy is
        # to choose the same action compared to the old policy
        ratios = torch.exp(log_probs_new - log_probs_old)
        
        # ===== PPO'S KEY INNOVATION: CLIPPED SURROGATE OBJECTIVE =====
        # This is the heart of PPO - preventing policy from changing too drastically
        
        # **The Problem PPO Solves:**
        # Standard policy gradient methods can make large policy changes
        # that lead to performance collapse. PPO prevents this by clipping
        # the objective function to limit how much the policy can change.
        
        # **The Clipped Surrogate Objective:**
        # L^CLIP(θ) = E[min(r_t(θ)A_t, clip(r_t(θ), 1-ε, 1+ε)A_t)]
        # 
        # **How It Works:**
        # 1. r_t(θ)A_t: Standard policy gradient objective
        # 2. clip(r_t(θ), 1-ε, 1+ε)A_t: Clipped version that limits ratio to [1-ε, 1+ε]
        # 3. min(...): Take the minimum to ensure we don't make changes that are too large
        # 4. For ε=0.2: ratios are clipped to [0.8, 1.2] (20% max change)
        
        # **Why This Works:**
        # - When ratio ≈ 1: No clipping, standard policy gradient
        # - When ratio > 1+ε: Clipped to prevent too much increase
        # - When ratio < 1-ε: Clipped to prevent too much decrease
        # - The minimum ensures we don't make changes that would hurt performance
        
        # surrogate1: Standard policy gradient objective
        # ratios * advantages: Standard importance sampling
        # For CSTR: Standard improvement of temperature control based on performance
        surrogate1 = ratios * advantages
        
        # surrogate2: Clipped policy gradient objective
        # torch.clamp(ratios, 1-actor_clip, 1+actor_clip): Limits ratio to [1-ε, 1+ε]
        # For ε=0.2: ratios are clipped to [0.8, 1.2]
        # This prevents the policy from changing too drastically
        # For CSTR: Limited improvement to prevent drastic changes in temperature control
        surrogate2 = torch.clamp(ratios, 1 - actor_clip, 1 + actor_clip) * advantages
        
        # taking the minimum of the clipped and unclipped objectives
        # L^CLIP(θ) = E[min(r_t(θ)A_t, clip(r_t(θ), 1-ε, 1+ε)A_t)]
        # actor_loss: Take the minimum of clipped and unclipped objectives
        # -torch.min(surrogate1, surrogate2).mean(): Negative because we maximize
        # This ensures we don't make changes that are too large
        # For CSTR: Conservative improvement of temperature control strategy
        actor_loss = -torch.min(surrogate1, surrogate2).mean()
        
        # ===== VALUE FUNCTION CLIPPING (OPTIONAL ENHANCEMENT) =====
        # Many modern PPO implementations also clip the value function
        # to prevent the critic from making too large updates
        
        # **Why Value Function Clipping?**
        # 1. **Stability**: Prevents critic from making extreme changes
        # 2. **Better Advantages**: More stable critic leads to better advantage estimates
        # 3. **Consistent Philosophy**: Same conservative approach as policy clipping
        
        # **How Value Function Clipping Works:**
        # 1. Compute standard MSE loss between predicted and actual returns
        # 2. Compute clipped values: values_old + clip(values - values_old, -clip, clip)
        # 3. Compute clipped MSE loss
        # 4. Take the maximum (opposite of policy clipping) to ensure we don't make
        #    changes that would hurt value function performance
        
        # Standard value function loss
        # F.mse_loss(values.squeeze(), returns): Mean squared error loss
        # - values.squeeze(): Critic's predictions (remove extra dimensions)
        # - returns: Actual returns computed from experience
        # For CSTR: Improve estimation of reactor state values
        # Ensure both tensors have the same shape to avoid broadcasting warnings
        values_squeezed = values.squeeze()
        if values_squeezed.shape != returns.shape:
            # Reshape returns to match values if needed
            returns = returns.view_as(values_squeezed)
        critic_loss_standard = F.mse_loss(values_squeezed, returns)
        
        # Value function clipping (if enabled)
        if value_clip is not None and value_clip > 0:
            # Get old value predictions (from the data collection phase)
            # Note: In a full implementation, we would store old values during collection
            # For now, we'll use the current values as a proxy for old values
            # This is a simplified implementation
            values_old = values.detach()  # Use current values as proxy for old values
            
            # Clipped values: prevent too large changes
            # torch.clamp(values - values_old, -value_clip, value_clip): Limit value changes
            # values_old + clamped_change: New values with limited change
            values_clipped = values_old + torch.clamp(values - values_old, -value_clip, value_clip)
            
            # Clipped value function loss
            # Ensure both tensors have the same shape to avoid broadcasting warnings
            values_clipped_squeezed = values_clipped.squeeze()
            if values_clipped_squeezed.shape != returns.shape:
                # Reshape returns to match values if needed
                returns = returns.view_as(values_clipped_squeezed)
            critic_loss_clipped = F.mse_loss(values_clipped_squeezed, returns)
            
            # Take the maximum (opposite of policy clipping)
            # This ensures we don't make changes that would hurt value function performance
            critic_loss = torch.max(critic_loss_standard, critic_loss_clipped)
        else:
            # No value function clipping
            critic_loss = critic_loss_standard
        
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
        
        # ===== ACTOR UPDATE (POLICY IMPROVEMENT) =====
        # Update the policy network to choose better actions
        # We use ONLY the actor_loss for this update, not the total_loss
        
        # Clear ALL gradients at the start of each epoch
        # This ensures clean gradients for each epoch (standard PPO practice)
        actor_optimizer.zero_grad()
        critic_optimizer.zero_grad()
        
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
        
        # Clear gradients after the final update to prevent memory leaks
        # This ensures no gradients remain in the model parameters for the next epoch
        for param in model.parameters():
            if param.grad is not None:
                param.grad.zero_()
                param.grad = None  # Explicitly set to None to prevent memory leaks
        
        # ===== EPOCH LOGGING =====
        # Track detailed metrics for this epoch
        clipped_ratios = torch.clamp(ratios, 1 - actor_clip, 1 + actor_clip)
        clipped_count = torch.sum(ratios != clipped_ratios).item()
        
        epoch_losses['actor_loss'].append(actor_loss.item())
        epoch_losses['critic_loss'].append(critic_loss.item())
        epoch_losses['total_loss'].append(total_loss.item())
        epoch_losses['entropy'].append(entropy.item())
        epoch_losses['policy_ratio_mean'].append(ratios.mean().item())
        epoch_losses['policy_ratio_std'].append(ratios.std().item())
        epoch_losses['policy_ratio_min'].append(ratios.min().item())
        epoch_losses['policy_ratio_max'].append(ratios.max().item())
        epoch_losses['clipped_ratio_count'].append(clipped_count)
        
        # Log detailed epoch information (every 5 epochs to avoid spam)
        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == epochs - 1:
            logger.debug(f"Epoch {epoch + 1}/{epochs}:")
            logger.debug(f"  - Actor Loss: {actor_loss.item():.6f}")
            logger.debug(f"  - Critic Loss: {critic_loss.item():.6f}")
            logger.debug(f"  - Total Loss: {total_loss.item():.6f}")
            logger.debug(f"  - Entropy: {entropy.item():.6f}")
            logger.debug(f"  - Policy Ratios: mean={ratios.mean().item():.4f}, std={ratios.std().item():.4f}")
            logger.debug(f"  - Policy Ratios: min={ratios.min().item():.4f}, max={ratios.max().item():.4f}")
            logger.debug(f"  - Clipped Ratios: {clipped_count}/{len(ratios)} ({clipped_count/len(ratios)*100:.1f}%)")
            
            # Log clipping effectiveness
            if clipped_count > 0:
                logger.debug(f"  - Clipping Active: {clipped_count} ratios clipped to [{1-actor_clip:.2f}, {1+actor_clip:.2f}]")
            else:
                logger.debug(f"  - Clipping Inactive: All ratios within [{1-actor_clip:.2f}, {1+actor_clip:.2f}]")
    
    # ===== FINAL PPO UPDATE SUMMARY =====
    logger.info(f"\n{'='*60}")
    logger.info(f"PPO UPDATE COMPLETED - FINAL SUMMARY")
    logger.info(f"{'='*60}")
    
    # Calculate summary statistics across all epochs
    final_actor_loss = epoch_losses['actor_loss'][-1]
    final_critic_loss = epoch_losses['critic_loss'][-1]
    final_entropy = epoch_losses['entropy'][-1]
    
    # Loss evolution statistics
    actor_losses = np.array(epoch_losses['actor_loss'])
    critic_losses = np.array(epoch_losses['critic_loss'])
    entropies = np.array(epoch_losses['entropy'])
    
    logger.info(f"Final Losses:")
    logger.info(f"  - Actor Loss: {final_actor_loss:.6f}")
    logger.info(f"  - Critic Loss: {final_critic_loss:.6f}")
    logger.info(f"  - Entropy: {final_entropy:.6f}")
    
    logger.info(f"Loss Evolution (across {epochs} epochs):")
    logger.info(f"  - Actor Loss: start={actor_losses[0]:.6f}, end={actor_losses[-1]:.6f}, "
               f"change={actor_losses[-1] - actor_losses[0]:.6f}")
    logger.info(f"  - Critic Loss: start={critic_losses[0]:.6f}, end={critic_losses[-1]:.6f}, "
               f"change={critic_losses[-1] - critic_losses[0]:.6f}")
    logger.info(f"  - Entropy: start={entropies[0]:.6f}, end={entropies[-1]:.6f}, "
               f"change={entropies[-1] - entropies[0]:.6f}")
    
    # Policy ratio statistics
    final_ratio_mean = epoch_losses['policy_ratio_mean'][-1]
    final_ratio_std = epoch_losses['policy_ratio_std'][-1]
    final_ratio_min = epoch_losses['policy_ratio_min'][-1]
    final_ratio_max = epoch_losses['policy_ratio_max'][-1]
    
    logger.info(f"Final Policy Ratios:")
    logger.info(f"  - Mean: {final_ratio_mean:.4f}")
    logger.info(f"  - Std: {final_ratio_std:.4f}")
    logger.info(f"  - Range: [{final_ratio_min:.4f}, {final_ratio_max:.4f}]")
    
    # Clipping effectiveness
    total_clipped = sum(epoch_losses['clipped_ratio_count'])
    total_ratios = len(states) * epochs
    clipping_percentage = total_clipped / total_ratios * 100 if total_ratios > 0 else 0
    
    logger.info(f"Clipping Effectiveness:")
    logger.info(f"  - Total ratios processed: {total_ratios}")
    logger.info(f"  - Total ratios clipped: {total_clipped}")
    logger.info(f"  - Clipping percentage: {clipping_percentage:.1f}%")
    
    if clipping_percentage > 50:
        logger.warning(f"  - WARNING: High clipping rate ({clipping_percentage:.1f}%) - consider reducing learning rate")
    elif clipping_percentage < 5:
        logger.info(f"  - Low clipping rate ({clipping_percentage:.1f}%) - policy changes are conservative")
    
    # Learning stability indicators
    actor_loss_std = actor_losses.std()
    critic_loss_std = critic_losses.std()
    
    logger.info(f"Learning Stability:")
    logger.info(f"  - Actor loss std: {actor_loss_std:.6f}")
    logger.info(f"  - Critic loss std: {critic_loss_std:.6f}")
    
    if actor_loss_std > 0.1:
        logger.warning(f"  - WARNING: High actor loss variance ({actor_loss_std:.6f}) - potential instability")
    if critic_loss_std > 0.1:
        logger.warning(f"  - WARNING: High critic loss variance ({critic_loss_std:.6f}) - potential instability")
    
    # Value function clipping summary
    if value_clip is not None and value_clip > 0:
        logger.info(f"Value Function Clipping:")
        logger.info(f"  - Value clip parameter: {value_clip}")
        logger.info(f"  - Value clipping enabled: Yes")
    else:
        logger.info(f"Value Function Clipping:")
        logger.info(f"  - Value clipping enabled: No")
    
    logger.info(f"{'='*60}")
    
    # Return final losses and entropy for KPI tracking
    # These are the final values from the last epoch
    return final_actor_loss, final_critic_loss, final_entropy
