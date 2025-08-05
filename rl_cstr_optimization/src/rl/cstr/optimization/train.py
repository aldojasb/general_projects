"""
Training script for PPO/DDPG agents.

This module provides:
- Main training loop
- Agent training orchestration
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from rl.cstr.optimization.base_agent import (ActorCriticNet,
collect_trajectories,
compute_gae,
ppo_update
)
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
# get state and action dimensions
state_dim = env.observation_space.shape[0]  # [Ca, Cb, T]
action_dim = env.action_space.shape[0]      # cooling jacket temp adjustment


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


# ===== MAIN TRAINING LOOP: COMPLETE PPO ALGORITHM =====
# This is where all the PPO components come together to create a complete
# reinforcement learning system for CSTR optimization
#
# **PPO Algorithm Overview:**
# PPO follows a clear iterative loop that balances exploration, learning, and stability:
# 1. **Collect Experience** (Rollouts) → 2. **Compute Advantages** (GAE) → 3. **Update Policy** (Clipped Objective)
#
# **For CSTR Context:**
# - **Rollouts**: Test current temperature control strategy in reactor
# - **Advantages**: Evaluate how well each temperature adjustment performed
# - **Updates**: Improve temperature control strategy based on performance
#
# **Key PPO Principles Implemented:**
# - **Proximal Policy Optimization**: Prevents drastic policy changes
# - **Actor-Critic Architecture**: Separate policy and value learning
# - **Generalized Advantage Estimation**: Stable advantage computation
# - **Multiple Epochs**: Efficient use of collected experience
# - **Separate Optimizers**: Independent control of policy and value learning

# ===== TRAINING CONFIGURATION =====
# num_updates: Total number of PPO update cycles
# Each update: collect data → compute advantages → update policy
# For CSTR: 1000 updates = 1000 cycles of temperature control improvement
num_updates = 1000

# ===== MAIN PPO TRAINING LOOP =====
# This loop implements the complete PPO algorithm
# Each iteration represents one complete cycle of the PPO algorithm
for update in range(num_updates):
    
    # ===== STEP 1: EXPERIENCE COLLECTION (ROLLOUTS) =====
    # Collect trajectories using the current policy
    # This is the "data collection" phase of PPO
    # For CSTR: Test current temperature control strategy in the reactor
    # Returns: states, actions, rewards, dones, values, log_probs_old
    # - states: Reactor conditions [Ca, Cb, T] at each timestep
    # - actions: Temperature adjustments applied at each timestep
    # - rewards: Conversion efficiency and safety rewards received
    # - dones: Whether reactor reached unsafe conditions or time limit
    # - values: Critic's predictions of expected future rewards
    # - log_probs_old: Action probabilities under the current policy
    states, actions, rewards, dones, values, log_probs_old = collect_trajectories(
        model, env, steps=2048)
    
    # ===== STEP 2: ADVANTAGE COMPUTATION (GAE) =====
    # Compute advantages using Generalized Advantage Estimation
    # This is the "learning signal" phase of PPO
    # For CSTR: Evaluate how much better/worse each temperature adjustment was than expected
    # Returns: advantages, returns
    # - advantages: How much better/worse actions were than expected (normalized)
    # - returns: Total expected future rewards from each state
    advantages, returns = compute_gae(rewards, dones, values)
    
    # ===== STEP 3: POLICY AND VALUE FUNCTION UPDATE =====
    # Update both the policy (actor) and value function (critic)
    # This is the "learning" phase of PPO
    # For CSTR: Improve temperature control strategy and reactor state estimation
    # - model: Current actor-critic network
    # - states: Reactor conditions from collected experience
    # - actions: Temperature adjustments from collected experience
    # - log_probs_old: Action probabilities under old policy (for importance sampling)
    # - returns: Total future rewards (for critic learning)
    # - advantages: How much better/worse actions were (for actor learning)
    ppo_update(model, states, actions, log_probs_old, returns, advantages)
    
    # ===== PROGRESS MONITORING =====
    # Print progress every 100 updates
    # This helps track training progress and identify potential issues
    # For CSTR: Monitor temperature control strategy improvement over time
    if (update + 1) % 100 == 0:
        print(f"Update {update + 1}/{num_updates} completed.")

# ===== MODEL PERSISTENCE =====
# Save the trained actor-critic model after training
# This preserves the learned policy and value function for later use
# For CSTR: Save the optimized temperature control strategy
# torch.save(model.state_dict(), "ppo_actor_critic_cstr.pth"):
# - model.state_dict(): Extracts all network parameters (weights and biases)
# - "ppo_actor_critic_cstr.pth": File path to save the model
# - Both actor and critic parameters are saved together
torch.save(model.state_dict(), "ppo_actor_critic_cstr.pth")  # clearly storing both actor and critic parameters
