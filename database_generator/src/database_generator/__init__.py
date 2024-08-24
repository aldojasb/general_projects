# __init__.py
# __init__.py

from .initializer import initialize_anomaly_detection

# Initialize the package
timestamp_for_this_experiment = initialize_anomaly_detection()

# Make timestamp_for_this_experiment available package-wide
__all__ = ['timestamp_for_this_experiment']