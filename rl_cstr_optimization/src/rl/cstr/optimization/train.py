"""
Training script for PPO agents.

This module provides:
- Main training loop
- Agent training orchestration
- Comprehensive KPI tracking
- Model checkpointing and early stopping
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
from rl.cstr.optimization.load_config_files import load_and_create_env_params
import os
from datetime import datetime
import logging
from dotenv import load_dotenv
# Import logger_manager
from logger.manager.logging_config import setup_logging_for_this_script

# ============================================================================
# LOGGING SETUP
# ============================================================================

# Load the .env file only if it exists
dotenv_path = '/workspace/general_projects/rl_cstr_optimization/.env'
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)
    print(f"Loaded environment variables from {dotenv_path}")
else:
    print(f"No .env file found at {dotenv_path}, relying on system environment variables.")

# Access the environment variable, with a fallback
path_to_logs = os.getenv('PATH_TO_SAVE_THE_LOGS')
print(f"Logs will be saved to: {path_to_logs}")
# -

# Set up logging for this script
# This will create logs in the directory specified by PATH_TO_SAVE_THE_LOGS environment variable
setup_logging_for_this_script(log_level=logging.INFO)

# Create a logger for this module
logger = logging.getLogger(__name__)

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

logger.info(f"Configuration loaded: {env_params}")
logger.info(f"Action space: {a_space}")
logger.info(f"Observation space: {o_space}")
logger.info(f"Number of steps: {nsteps}")


# Create the CSTR environment with proper parameters
# The environment simulates a continuously stirred tank reactor for chemical process control
env = make_env(env_params)
# get state and action dimensions
state_dim = env.observation_space.shape[0]  # [Ca, Cb, T]
action_dim = env.action_space.shape[0]      # cooling jacket temp adjustment


# ============================================================================
# MODEL INITIALIZATION
# ============================================================================

# Initialize the actor-critic network
model = ActorCriticNet(state_dim, action_dim)

# ============================================================================
# OPTIMIZER INITIALIZATION
# ============================================================================

# We use TWO separate optimizers for actor and critic networks
# This is a key design decision in PPO for training stability and control
# Why separate actor optimizer?
# 1. **Different learning objectives**: Actor learns policy, critic learns value function
# 2. **Different learning rates**: Actor often needs slower learning for stability
# 3. **Independent updates**: Prevents one network from interfering with the other
# 4. **Gradient control**: Can apply different gradient clipping/regularization

# Why separate critic optimizer?
# 1. **Faster convergence**: Value functions often converge faster than policies
# 2. **Different loss functions**: MSE for critic vs. policy gradient for actor
# 3. **Stability**: Prevents critic updates from destabilizing policy learning
# 4. **Independent momentum**: Adam's momentum states are separate for each network


# ===== ACTOR OPTIMIZER =====
# Optimizes the policy network (actor) parameters
# - model.actor.parameters(): Shared feature extraction layers
# - model.mean_head.weight/bias: Action mean prediction layer
# - model.log_std: Learnable standard deviation parameter
# - Learning rate: 3e-4 (typical for policy optimization)

actor_optimizer = optim.Adam([
    {'params': model.actor.parameters()},  # Shared actor layers
    {'params': [model.mean_head.weight, model.mean_head.bias, model.log_std]}  # Policy output parameters
], lr=3e-4)

# ===== CRITIC OPTIMIZER =====
# Optimizes the value function network (critic) parameters
# - model.critic.parameters(): All critic network layers
# - Learning rate: 1e-3 (faster than actor for accurate value estimation)
#

critic_optimizer = optim.Adam(model.critic.parameters(), lr=1e-3)

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

# Training hyperparameters
num_updates = 1000
steps_per_update = 2048
early_stopping_patience = 200  # Stop if no improvement for 200 updates
min_improvement = 0.01         # Minimum improvement threshold

# Training state
best_reward = float('-inf')
patience_counter = 0

# Training timing and metrics
import time
training_start_time = time.time()
avg_time_per_update = 0.0

# path to save the best model
best_model_path = os.path.join("/workspace/general_projects/rl_cstr_optimization/trained_models", "best_model.pth")

logger.info(f"Training configuration:")
logger.info(f"  - Total updates: {num_updates}")
logger.info(f"  - Steps per update: {steps_per_update}")
logger.info(f"  - Early stopping patience: {early_stopping_patience}")

# ============================================================================
# MAIN TRAINING LOOP: COMPLETE PPO ALGORITHM
# ============================================================================

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

logger.info(f"\n{'='*60}")
logger.info(f"STARTING PPO TRAINING FOR CSTR OPTIMIZATION")
logger.info(f"{'='*60}")
logger.info(f"Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

try:
    # ===== MAIN PPO TRAINING LOOP =====
    # This loop implements the complete PPO algorithm
    # Each iteration represents one complete cycle of the PPO algorithm
    for update in range(num_updates):
        
        # Record start time for this update
        update_start_time = time.time()
        
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
        
        
        states, actions, rewards, terminated_list, truncated_list, values, log_probs_old = collect_trajectories(
            model=model,
            env=env,
            steps=steps_per_update)
        
        # ===== STEP 2: ADVANTAGE COMPUTATION (GAE) =====
        # Compute advantages using Generalized Advantage Estimation
        # This is the "learning signal" phase of PPO
        # For CSTR: Evaluate how much better/worse each temperature adjustment was than expected
        # Returns: gae_advantages_normalized, total_expected_future_rewards, raw_gae_advantages
        # - gae_advantages_normalized: How much better/worse actions were than expected (normalized)
        # - total_expected_future_rewards: Total expected future rewards from each state
        # - raw_gae_advantages: Raw advantage estimates for each action
        gae_advantages_normalized, total_expected_future_rewards, raw_gae_advantages = compute_gae(
            rewards=rewards,
            dones=dones,
            values=values)
        
        # ===== STEP 3: POLICY AND VALUE FUNCTION UPDATE =====
        # Update both the policy (actor) and value function (critic)
        # This is the "learning" phase of PPO
        # For CSTR: Improve temperature control strategy and reactor state estimation
        # - model: Current actor-critic network
        # - states: Reactor conditions from collected experience
        # - actions: Temperature adjustments from collected experience
        # - log_probs_old: Action probabilities under old policy (for importance sampling)
        # - total_expected_future_rewards: Total future rewards (for critic learning)
        # - gae_advantages_normalized: How much better/worse actions were (for actor learning)
        
        # Get losses from PPO update for KPI tracking
        policy_loss, value_loss, entropy = ppo_update(model,
        states=states,
        actions=actions,
        log_probs_old=log_probs_old,
        returns=total_expected_future_rewards,
        advantages=gae_advantages_normalized,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
        actor_clip=0.2,
        value_clip=0.2,
        epochs=50
        )
        
        # Calculate update time and update average
        update_time = time.time() - update_start_time
        avg_time_per_update = (avg_time_per_update * update + update_time) / (update + 1)
        
        # ===== IMPROVED PROGRESS MONITORING =====
        # IMPROVEMENT 1: More frequent progress updates (every 10 updates instead of 100)
        if (update + 1) % 10 == 0:
            logger.info(f"Update {update + 1}/{num_updates} completed.")
            logger.info(f"   Current reward: {current_reward:.4f}")
            logger.info(f"   Best reward: {best_reward:.4f}")
            logger.info(f"   Patience counter: {patience_counter}")
        
        # IMPROVEMENT 2: Training metrics with detailed reward tracking
        current_reward = np.mean(rewards) if rewards else 0.0
        
        # Log detailed metrics every 5 updates for better visibility
        if (update + 1) % 5 == 0:
            logger.info(f"Update {update + 1}: Reward={current_reward:.6f}, Best={best_reward:.6f}, Patience={patience_counter}")
            logger.info(f"   Policy Loss: {policy_loss:.6f}, Value Loss: {value_loss:.6f}, Entropy: {entropy:.6f}")
            logger.info(f"   Update Time: {update_time:.2f}s, Avg Time: {avg_time_per_update:.2f}s")
        
        # IMPROVEMENT 3: Convergence warnings
        if patience_counter > 50:
            logger.warning(f"No improvement for {patience_counter} updates - consider adjusting learning rates")
        if patience_counter > 100:
            logger.warning(f"Training may be stuck - patience counter: {patience_counter}/{early_stopping_patience}")
        
        # IMPROVEMENT 4: Time estimates
        if (update + 1) % 20 == 0:
            updates_remaining = num_updates - (update + 1)
            estimated_time_remaining = updates_remaining * avg_time_per_update
            estimated_minutes = estimated_time_remaining / 60
            logger.info(f"Progress: {(update + 1)/num_updates*100:.1f}% complete")
            logger.info(f"Estimated time remaining: {estimated_minutes:.1f} minutes")
        
        # ===== EARLY STOPPING AND CHECKPOINTING =====
        # Check if we should save the best model
        if current_reward > best_reward + min_improvement:
            best_reward = current_reward
            patience_counter = 0
            
            # Save best model
            torch.save({
                'model_state_dict': model.state_dict(),
                'actor_optimizer_state_dict': actor_optimizer.state_dict(),
                'critic_optimizer_state_dict': critic_optimizer.state_dict(),
                'update': update,
                'best_reward': best_reward,
                'training_config': {
                    'num_updates': num_updates,
                    'steps_per_update': steps_per_update,
                    'actor_lr': 3e-4,
                    'critic_lr': 1e-3
                }
            }, best_model_path)
            
            logger.info(f"New best model saved! Reward: {best_reward:.6f}")
        else:
            patience_counter += 1
        
        # Check for early stopping
        if patience_counter >= early_stopping_patience:
            logger.warning(f"\n Early stopping triggered after {update + 1} updates")
            logger.warning(f"   No improvement for {early_stopping_patience} updates")
            logger.warning(f"   Best reward achieved: {best_reward:.6f}")
            break
        
        # ===== PROGRESS MONITORING =====
        # Print progress every 100 updates
        # This helps track training progress and identify potential issues
        # For CSTR: Monitor temperature control strategy improvement over time
        if (update + 1) % 100 == 0:
            logger.info(f" Update {update + 1}/{num_updates} completed.")
            logger.info(f"   Current reward: {current_reward:.2f}")
            logger.info(f"   Best reward: {best_reward:.2f}")
            logger.info(f"   Patience counter: {patience_counter}")

except KeyboardInterrupt:
    logger.warning(f"\n  Training interrupted by user at update {update}")
except Exception as e:
    logger.error(f"\n Training error at update {update}: {str(e)}")
    raise
finally:
    # ===== FINAL SAVE AND SUMMARY =====
    # Save the final model and training summary
    torch.save({
        'model_state_dict': model.state_dict(),
        'actor_optimizer_state_dict': actor_optimizer.state_dict(),
        'critic_optimizer_state_dict': critic_optimizer.state_dict(),
        'update': update,
        'best_reward': best_reward,
        'training_config': {
            'num_updates': num_updates,
            'steps_per_update': steps_per_update,
            'actor_lr': 3e-4,
            'critic_lr': 1e-3
        }
    }, best_model_path)
    
    # Calculate total training time
    total_training_time = time.time() - training_start_time
    total_minutes = total_training_time / 60
    
    # Print final summary
    logger.info(f"\n{'='*60}")
    logger.info(f"TRAINING COMPLETED - FINAL SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Total updates completed: {update + 1}")
    logger.info(f"Best mean reward: {best_reward:.6f}")
    logger.info(f"Total training time: {total_minutes:.1f} minutes")
    logger.info(f"Average time per update: {avg_time_per_update:.2f} seconds")
    logger.info(f"Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"{'='*60}")

logger.info(f"\n Training script completed successfully!")
logger.info(f" Training logs saved to: training_logs/")
