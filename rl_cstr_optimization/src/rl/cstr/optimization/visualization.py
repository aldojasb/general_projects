"""
Plotting functions for KPIs (Key Performance Indicators) in RL simulations.

This module contains functions for creating interactive visualizations
of simulation results using Plotly.
"""

from typing import Any, Optional, Union
import numpy as np
import plotly.graph_objects as go


def plot_state_variables(
    observations: np.ndarray, 
    variable_definitions: list[dict[str, Any]]
) -> None:
    """
    Create separate plots for each state variable over time.
    
    Args:
        observations: Array of observations
        variable_definitions: List of dictionaries defining each variable's properties.
            Each dict should contain:
            - 'name': str - Display name for the variable
            - 'index': int - Column index in observations array
            - 'color': str - Color for the plot line
            - 'symbol': str - Marker symbol
            - 'yaxis_title': str - Y-axis title
            
            Example:
            [
                {
                    'name': 'Ca (Concentration A)',
                    'index': 0,
                    'color': 'blue',
                    'symbol': 'circle',
                    'yaxis_title': 'Concentration (mol/L)'
                },
                {
                    'name': 'T (Temperature)',
                    'index': 1,
                    'color': 'red',
                    'symbol': 'square',
                    'yaxis_title': 'Temperature (K)'
                }
            ]
            
            If None, defaults to CSTR variables (Ca, T, Cb)
    """
    # Create steps array for x-axis
    steps = list(range(1, len(observations) + 1))
    
    # Create separate plot for each variable
    for var in variable_definitions:
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=steps,
            y=observations[:, var['index']],
            mode='lines+markers',
            name=var['name'],
            line=dict(color=var['color'], width=3),
            marker=dict(size=8, symbol=var['symbol'])
        ))
        
        fig.update_layout(
            title=f'{var["name"]} Over Time',
            xaxis_title='Simulation Step',
            yaxis_title=var['yaxis_title'],
            template='plotly_white',
            height=400,
            showlegend=False
        )
        
        fig.show()


def plot_control_actions(actions: np.ndarray) -> None:
    """
    Create plot showing control actions over time.
    
    Args:
        actions: Array of actions (can be 1D or 2D)
    """
    # Create steps array for x-axis
    steps = list(range(1, len(actions) + 1))
    
    # Ensure actions is 2D for consistent processing
    if actions.ndim == 1:
        actions = actions.reshape(-1, 1)
    
    fig = go.Figure()
    
    # Add trace for the single action component
    action_names = ['Tc (Coolant Temperature)']
    colors = ['blue']
    symbols = ['circle']
    
    for i in range(actions.shape[1]):
        fig.add_trace(go.Scatter(
            x=steps,
            y=actions[:, i],
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


def plot_reward_evolution(rewards: np.ndarray) -> None:
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

