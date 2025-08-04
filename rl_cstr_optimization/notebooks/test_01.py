# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# +
"""
CSTR (Continuously Stirred Tank Reactor) Environment Demo
========================================================

This demo showcases the CSTR environment from pc-gym, which simulates
a continuously stirred tank reactor for chemical process control.

The CSTR is a well-established model that's thoroughly tested and stable,
making it perfect for learning and experimentation.

State Variables (3 total):
- Ca: Concentration of reactant A (mol/L)
- T: Temperature (K)
- Cb: Concentration of reactant B (mol/L)

Action Variables (1 total):
- Tc: Coolant temperature (K)

Observations (3 total):
- Ca, T, Cb: Concentration A, Temperature, Concentration B
"""

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

# Reset the environment to get initial state
# This returns the initial observation (concentrations at reactor outlet)
initial_observation, initial_info = env.reset()

print("=" * 60)
print(f"CSTR REACTOR DEMO - {nsteps} STEP SIMULATION")
print("=" * 60)
print(f"Initial observation (normalized): {initial_observation}")

# Denormalize the initial observation for better understanding
initial_real = denormalize_observations(initial_observation, o_space)

print(f"Initial observation (real values):")
print(f"  Ca: {initial_real[0]}")
print(f"  T:  {initial_real[1]}")
print(f"  Cb: {initial_real[2]}")

print(f"Observation shape: {initial_observation.shape}")
print(f"Action space: {env.action_space}")
print(f"Observation space: {env.observation_space}")
print("=" * 60)

# +
# ============================================================================
# DATA STORAGE FOR ANALYSIS
# ============================================================================

# Lists to store data for visualization and analysis
observations = [initial_observation]  # Store all observations (concentrations at outlet)
actions = []       # Store all actions taken
rewards = []       # Store all rewards received
states = []        # Store full state information if available
denorm_actions = [] # Store denormalized actions
denorm_observations = [] # Store denormalized observations

# +
# ============================================================================
# MAIN SIMULATION LOOP - nsteps STEPS
# ============================================================================

print(f"\nStarting {nsteps}-step simulation...")
print("-" * 40)

for step in range(nsteps):
    print(f"\nStep {step + 1}/{nsteps}:")

    # ========================================================================
    # ACTION SELECTION
    # ========================================================================
    
    # For this demo, we'll use a simple strategy:
    # - Moderate coolant temperature control
    # - Stay within the safe operating range
    
    # Since actions are normalized, we need to provide values between 0 and 1
    # These will be automatically scaled to the actual bounds defined in env_params
    action = np.array([
        0.5,    # Tc: Coolant temperature (normalized) - moderate value
    ])
    
    # Denormalize: actual_value = low + (normalized_value * (high - low))
    denorm_action = denormalize_actions(action, a_space)
    
    
    # ========================================================================
    # ENVIRONMENT STEP
    # ========================================================================
    
    # Execute the action in the environment
    # This advances the simulation by one time step
    # Returns: new_observation, reward, terminated, truncated, info
    observation, reward, terminated, truncated, info = env.step(action)
    
    # Denormalize observation for better understanding
    denorm_observation = denormalize_observations(observation, o_space)

    print(f" Normalized action: {action}")
    print(f" Denormalized action: {denorm_action}")
    print(f"  New observation (normalized): {observation}")
    print(f"  New observation (real values):")
    print(f"    Ca: {denorm_observation[0]}")
    print(f"    T:  {denorm_observation[1]}")
    print(f"    Cb: {denorm_observation[2]}")
    print(f"  Reward: {reward:.4f}")
    print(f"  Terminated: {terminated}")
    print(f"  Truncated: {truncated}")
    
    # ========================================================================
    # DATA STORAGE
    # ========================================================================
    
    # Store the data for later analysis
    observations.append(observation.copy())
    actions.append(action.copy())
    rewards.append(reward)
    denorm_actions.append(denorm_action.copy())
    denorm_observations.append(denorm_observation.copy())
    
    # ========================================================================
    # TERMINATION CHECK
    # ========================================================================
    
    # Check if the episode has ended
    if terminated or truncated:
        print(f"  Episode ended at step {step + 1}")
        break


# +
# ============================================================================
# SIMULATION COMPLETE - ANALYSIS AND VISUALIZATION
# ============================================================================

print("\n" + "=" * 60)
print("SIMULATION COMPLETE - ANALYSIS")
print("=" * 60)

# Convert lists to numpy arrays for easier analysis
observations = np.array(observations)
actions = np.array(actions)
rewards = np.array(rewards)
denorm_actions = np.array(denorm_actions)
denorm_observations = np.array(denorm_observations)

print(f"Total steps completed: {len(observations)}")
print(f"Average reward: {np.mean(rewards):.4f}")
print(f"Total reward: {np.sum(rewards):.4f}")
print(f"Denormalized actions: {denorm_actions}")
print(f"Denormalized observations: {denorm_observations}")

# +
# ============================================================================
# VISUALIZATION
# ============================================================================
# -

# Create all four plots
plot_state_variables(denorm_observations)

# Create all four plots
plot_control_actions(denorm_actions)


plot_reward_evolution(rewards)



# Close the environment to free resources
env.close()
