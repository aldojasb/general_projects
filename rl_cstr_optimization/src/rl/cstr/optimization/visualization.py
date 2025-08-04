"""
Plotting functions for KPIs (Key Performance Indicators) in RL simulations.

This module contains functions for creating interactive visualizations
of simulation results using Plotly.
"""

import numpy as np
import plotly.graph_objects as go


def plot_state_variables(observations):
    """
    Create plot showing CSTR state variables over time.
    
    Args:
        observations: Array of observations [Ca, T, Cb]
    """
    # Create steps array for x-axis
    steps = list(range(1, len(observations) + 1))
    
    fig = go.Figure()
    
    # Add traces for the CSTR state variables (Ca, T, Cb)
    fig.add_trace(go.Scatter(
        x=steps, 
        y=observations[:, 0],  # Ca (Concentration A)
        mode='lines+markers',
        name='Ca (Concentration A)',
        line=dict(color='blue', width=3),
        marker=dict(size=8)
    ))
    
    fig.add_trace(go.Scatter(
        x=steps, 
        y=observations[:, 1],  # T (Temperature)
        mode='lines+markers',
        name='T (Temperature)',
        line=dict(color='red', width=3),
        marker=dict(size=8, symbol='square')
    ))
    
    fig.add_trace(go.Scatter(
        x=steps, 
        y=observations[:, 2],  # Cb (Concentration B)
        mode='lines+markers',
        name='Cb (Concentration B)',
        line=dict(color='green', width=3),
        marker=dict(size=8, symbol='triangle-up')
    ))
    
    fig.update_layout(
        title='CSTR State Variables Over Time',
        xaxis_title='Simulation Step',
        yaxis_title='Concentration (mol/L) / Temperature (K)',
        template='plotly_white',
        height=500
    )
    
    fig.show()


def plot_control_actions(actions, action_bounds):
    """
    Create plot showing control actions over time.
    
    Args:
        actions: Array of normalized actions
        action_bounds: Dictionary with 'low' and 'high' bounds for actions
    """
    # Create steps array for x-axis
    steps = list(range(1, len(actions) + 1))
    
    # Denormalize actions for plotting
    denorm_actions = np.zeros_like(actions)
    
    for i in range(actions.shape[1]):
        denorm_actions[:, i] = action_bounds['low'][i] + actions[:, i] * (action_bounds['high'][i] - action_bounds['low'][i])
    
    fig = go.Figure()
    
    # Add trace for the single action component
    action_names = ['Tc (Coolant Temperature)']
    colors = ['blue']
    symbols = ['circle']
    
    for i in range(denorm_actions.shape[1]):
        fig.add_trace(go.Scatter(
            x=steps,
            y=denorm_actions[:, i],
            mode='lines+markers',
            name=action_names[i],
            line=dict(color=colors[i], width=3),
            marker=dict(size=8, symbol=symbols[i])
        ))
    
    fig.update_layout(
        title='Control Actions Over Time',
        xaxis_title='Simulation Step',
        yaxis_title='Coolant Temperature (K)',
        template='plotly_white',
        height=500
    )
    
    fig.show()


def plot_reward_evolution(rewards):
    """
    Create plot showing reward evolution over time.
    
    Args:
        rewards: Array of rewards received
    """
    # Create steps array for x-axis
    steps = list(range(1, len(rewards) + 1))
    
    # Calculate cumulative rewards
    cumulative_rewards = np.cumsum(rewards)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=steps,
        y=rewards,
        mode='lines+markers',
        name='Individual Reward',
        line=dict(color='blue', width=3),
        marker=dict(size=8)
    ))
    
    fig.add_trace(go.Scatter(
        x=steps,
        y=cumulative_rewards,
        mode='lines+markers',
        name='Cumulative Reward',
        line=dict(color='red', width=3),
        marker=dict(size=8, symbol='square')
    ))
    
    fig.update_layout(
        title='Reward Evolution',
        xaxis_title='Simulation Step',
        yaxis_title='Reward',
        template='plotly_white',
        height=500
    )
    
    fig.show()


def plot_summary_dashboard(observations, actions, rewards, action_bounds):
    """
    Create summary dashboard with key statistics.
    
    Args:
        observations: Array of observations [Ca, T, Cb]
        actions: Array of normalized actions
        rewards: Array of rewards received
        action_bounds: Dictionary with 'low' and 'high' bounds for actions
    """
    # Denormalize actions for summary statistics
    denorm_actions = np.zeros_like(actions)
    
    for i in range(actions.shape[1]):
        denorm_actions[:, i] = action_bounds['low'][i] + actions[:, i] * (action_bounds['high'][i] - action_bounds['low'][i])
    
    # Create summary statistics
    summary_stats = {
        'Metric': ['Total Steps', 'Average Reward', 'Total Reward', 'Min Reward', 'Max Reward',
                  'Final Ca (Concentration A)', 'Final T (Temperature)', 'Final Cb (Concentration B)',
                  'Avg Coolant Temperature'],
        'Value': [
            len(observations),
            f"{np.mean(rewards):.4f}",
            f"{np.sum(rewards):.4f}",
            f"{np.min(rewards):.4f}",
            f"{np.max(rewards):.4f}",
            f"{observations[-1, 0]:.4f} mol/L",  # Ca (Concentration A)
            f"{observations[-1, 1]:.4f} K",      # T (Temperature)
            f"{observations[-1, 2]:.4f} mol/L",  # Cb (Concentration B)
            f"{np.mean(denorm_actions[:, 0]):.2f} K"
        ]
    }
    
    # Create summary table
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=['Metric', 'Value'],
            fill_color='lightblue',
            align='left',
            font=dict(size=14, color='black')
        ),
        cells=dict(
            values=[summary_stats['Metric'], summary_stats['Value']],
            fill_color='white',
            align='left',
            font=dict(size=12)
        )
    )])
    
    fig.update_layout(
        title='Simulation Summary',
        height=400
    )
    
    fig.show()
