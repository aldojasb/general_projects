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
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pcgym import make_env
from visualization.plot_kpis import plot_state_variables, plot_control_actions, plot_reward_evolution, plot_summary_dashboard

def create_cstr_demo():
    """
    Main function to demonstrate the CSTR reactor environment.
    This function creates the environment, runs a 10-step simulation,
    and visualizes the results.
    """
    
    # ============================================================================
    # ENVIRONMENT SETUP
    # ============================================================================
    
    # Define simulation parameters for CSTR reactor
    T = 26  # Simulation time (hours)
    nsteps = 10  # Number of steps for our demo
    
    # Define action space bounds for CSTR reactor
    # Action space: [Tc] - Coolant temperature (K)
    # Based on the example from pc-gym documentation
    a_space = {
        'low': np.array([295]),  # Lower bound for coolant temperature (K)
        'high': np.array([302])  # Upper bound for coolant temperature (K)
    }
    
    # Define observation space bounds for CSTR reactor
    # Observations: [Ca, T, Cb] - Concentration A, Temperature, Concentration B
    # Based on the example from pc-gym documentation
    o_space = {
        'low': np.array([0.7, 300, 0.8]),   # Lower bounds: [Ca_min, T_min, Cb_min]
        'high': np.array([1.0, 350, 0.9])   # Upper bounds: [Ca_max, T_max, Cb_max]
    }
    
    # Define initial conditions for CSTR reactor
    # Initial state: [Ca, T, Cb] - Concentration A, Temperature, Concentration B
    # Based on the example from pc-gym documentation
    x0 = np.array([0.8, 330, 0.8])  # Initial conditions: [Ca_0, T_0, Cb_0]
    
    # Define set points for the CSTR reactor
    # Based on the example from pc-gym documentation
    # We'll create a set point profile for Ca (concentration A) with varying targets
    # Make sure we have exactly nsteps elements
    if nsteps >= 3:
        # Create a profile similar to the original example
        third = nsteps // 3
        SP = {
            'Ca': [0.85 for i in range(third)] + 
                  [0.9 for i in range(third)] + 
                  [0.87 for i in range(nsteps - 2*third)],  # Target concentration A (mol/L)
        }
    else:
        # For small nsteps, use constant set point
        SP = {
            'Ca': [0.85 for i in range(nsteps)],  # Target concentration A (mol/L) - constant set point
        }
    
    # Define reward scaling for CSTR reactor
    # Based on the example from pc-gym documentation
    r_scale = {'Ca': 1e3}  # Reward scaling for concentration A
    
    # Create environment parameters dictionary
    # Based on the example from pc-gym documentation
    env_params = {
        'N': nsteps,                    # Number of steps
        'tsim': T,                      # Simulation time
        'SP': SP,                       # Set points for control objectives
        'o_space': o_space,             # Observation space bounds (3D)
        'a_space': a_space,             # Action space bounds (1D)
        'x0': x0,                      # Initial conditions (3D)
        'r_scale': r_scale,            # Reward scaling
        'model': 'cstr',                # Model type - CSTR is well-tested
        'normalise_a': True,            # Normalize actions
        'normalise_o': True,            # Normalize observations
        'noise': True,                  # Enable noise for realism
        'integration_method': 'casadi', # Use CasADi integration (stable for CSTR)
        'noise_percentage': 0.001,      # Noise percentage
    }
    
    # Create the CSTR environment with proper parameters
    # The environment simulates a continuously stirred tank reactor for chemical process control
    env = make_env(env_params)
    
    # Reset the environment to get initial state
    # This returns the initial observation (concentrations at reactor outlet)
    initial_observation, initial_info = env.reset()
    
    print("=" * 60)
    print("CSTR REACTOR DEMO - 10 STEP SIMULATION")
    print("=" * 60)
    print(f"Initial observation: {initial_observation}")
    print(f"Observation shape: {initial_observation.shape}")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    print("=" * 60)
    
    # ============================================================================
    # DATA STORAGE FOR ANALYSIS
    # ============================================================================
    
    # Lists to store data for visualization and analysis
    observations = []  # Store all observations (concentrations at outlet)
    actions = []       # Store all actions taken
    rewards = []       # Store all rewards received
    states = []        # Store full state information if available
    
    # ============================================================================
    # MAIN SIMULATION LOOP - 10 STEPS
    # ============================================================================
    
    print("\nStarting 10-step simulation...")
    print("-" * 40)
    
    for step in range(10):
        print(f"\nStep {step + 1}/10:")
        
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
        
        # Alternative: Random action within normalized bounds
        # action = np.random.uniform(0, 1, 1)
        
        # Calculate denormalized action values for display
        # Denormalize: actual_value = low + (normalized_value * (high - low))
        denorm_action = np.array([
            a_space['low'][0] + action[0] * (a_space['high'][0] - a_space['low'][0]),  # Tc
        ])
        
        print(f"  Normalized action: {action}")
        print(f"  Denormalized action: {denorm_action}")
        print(f"  - Coolant temperature: {denorm_action[0]:.2f} K")
        
        # ========================================================================
        # ENVIRONMENT STEP
        # ========================================================================
        
        # Execute the action in the environment
        # This advances the simulation by one time step
        # Returns: new_observation, reward, terminated, truncated, info
        observation, reward, terminated, truncated, info = env.step(action)
        
        print(f"  New observation: {observation}")
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
        
        # ========================================================================
        # TERMINATION CHECK
        # ========================================================================
        
        # Check if the episode has ended
        if terminated or truncated:
            print(f"  Episode ended at step {step + 1}")
            break
    
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
    
    print(f"Total steps completed: {len(observations)}")
    print(f"Average reward: {np.mean(rewards):.4f}")
    print(f"Total reward: {np.sum(rewards):.4f}")
    
    # ============================================================================
    # VISUALIZATION
    # ============================================================================
    
    # Create a comprehensive visualization of the results
    create_visualization(observations, actions, rewards, a_space)
    
    # ============================================================================
    # ENVIRONMENT CLEANUP
    # ============================================================================
    
    # Close the environment to free resources
    env.close()
    
    print("\nDemo completed successfully!")
    return observations, actions, rewards

def create_visualization(observations, actions, rewards, action_bounds):
    """
    Create comprehensive visualizations of the simulation results using Plotly.
    
    Args:
        observations: Array of observations (concentrations at reactor outlet)
        actions: Array of actions taken
        rewards: Array of rewards received
        action_bounds: Dictionary with 'low' and 'high' bounds for actions
    """
    
    # Create all four plots
    plot_state_variables(observations)
    plot_control_actions(actions, action_bounds)
    plot_reward_evolution(rewards)
    plot_summary_dashboard(observations, actions, rewards, action_bounds)
    
    print("\nInteractive visualizations created and displayed!")


def explain_environment_components():
    """
    Print detailed explanation of the CSTR environment components.
    This helps understand the system dynamics and control objectives.
    """
    
    print("\n" + "=" * 60)
    print("CSTR ENVIRONMENT EXPLANATION")
    print("=" * 60)
    
    print("""
    SYSTEM OVERVIEW:
    ================
    The CSTR (Continuously Stirred Tank Reactor) is a well-established
    chemical process control system that simulates:
    
    1. Continuous chemical reaction in a stirred tank
    2. Temperature control through coolant
    3. Concentration control of reactants
    
    STATE VARIABLES (3 total):
    ==========================
    - Ca: Concentration of reactant A (mol/L)
    - T: Temperature (K)
    - Cb: Concentration of reactant B (mol/L)
    
    ACTION VARIABLES (1 total):
    ===========================
    - Tc: Coolant temperature (295-302 K)
    
    OBSERVATIONS (3 total):
    ======================
    - Ca: Concentration of reactant A (mol/L)
    - T: Temperature (K)
    - Cb: Concentration of reactant B (mol/L)
    
    CONTROL OBJECTIVES:
    ===================
    1. Maintain target concentration of reactant A (Ca)
    2. Control temperature within safe operating range
    3. Ensure stable reaction conditions
    4. Handle disturbances and variations
    
    CHEMICAL REACTIONS:
    ===================
    The CSTR typically involves exothermic reactions where:
    - Reactant A is consumed to produce Reactant B
    - Temperature affects reaction rate
    - Coolant temperature controls heat removal
    
    This is a well-tested model that's perfect for learning
    process control and reinforcement learning concepts.
    """)

if __name__ == "__main__":
    """
    Main execution block.
    This runs the demo when the script is executed directly.
    """
    
    # Print environment explanation first
    explain_environment_components()
    
    # Run the main demo
    observations, actions, rewards = create_cstr_demo()
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("The demo has demonstrated:")
    print("1. Environment creation and setup")
    print("2. 10-step simulation with controlled actions")
    print("3. Data collection and analysis")
    print("4. Visualization of results")
    print("5. Understanding of system dynamics")
    print("=" * 60)
