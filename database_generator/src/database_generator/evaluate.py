import plotly.graph_objs as go

import plotly.graph_objs as go
import pandas as pd
import os


# Get the logger for this module
import logging
from database_generator.logging_configuration import setup_logging_for_this_script
setup_logging_for_this_script()
# Get the logger for this module
logger = logging.getLogger(__name__)

def overlaid_plots_with_plotly(df: pd.DataFrame,
                               scatter_variables: list = None,
                               variable_of_interest: str = None,
                               save_plot: bool = True,
                               save_path: str = 'tmp/',
                               filename: str = 'overlaid_overview') -> go.Figure:
    """
    Creates an interactive Plotly plot overlaying all columns of a DataFrame, with specified columns as scatter plots and an optional secondary y-axis.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing time series data.
    scatter_variables (list): List of column names to be displayed as scatter plots.
    variable_of_interest (str): Column name to be displayed on the secondary y-axis.
    save_plot (bool): Whether to save the plot as an HTML file. Defaults to True.
    save_path (str): Directory path where the plot will be saved. Defaults to 'tmp/'.
    filename (str): The filename for the saved plot. Defaults to 'overlaid_overview'.

    Returns:
    go.Figure: A Plotly Figure object containing the interactive plot with all data series.
    """
    # Initialize the Plotly figure
    fig = go.Figure()

    # Add scatter plots if scatter variables are specified
    if scatter_variables:
        for scatter_var in scatter_variables:
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df[scatter_var],
                mode='markers',
                marker=dict(size=8),
                name=scatter_var
            ))

    # Add line plots for the remaining variables except the variable_of_interest and scatter_variables
    for column in df.columns:
        if (scatter_variables is None or column not in scatter_variables) and column != variable_of_interest:
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df[column],
                mode='lines',
                name=column,
                yaxis='y1'
            ))

    # Add the variable_of_interest on the secondary y-axis if specified
    if variable_of_interest:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df[variable_of_interest],
            mode='lines',
            name=variable_of_interest,
            yaxis='y2',
            # line=dict(color='blue')  # Customize the line for distinction
        ))

        # Update the layout to include a secondary y-axis with matching scale
        fig.update_layout(
            yaxis2=dict(
                title=variable_of_interest,
                overlaying='y',
                side='right',
                matches='y'  # Ensure the secondary y-axis matches the scale of the primary y-axis
            )
        )

    # Update layout for better visualization
    fig.update_layout(
        title="Overlaid Overview",
        xaxis_title="Time",
        yaxis_title="Primary Axis",
        legend_title="Variables",
        width=1300,
        height=650,
        hovermode="x unified",
        legend=dict(
            x=1.08,  # Position the legend outside the plot area to the right
            y=1,  # Align it at the top
            xanchor='left',
            yanchor='top'
        )
    )

    filename = filename + '.html'
    # Save the plot if required
    if save_plot:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        fig.write_html(os.path.join(save_path, filename))

    return fig
