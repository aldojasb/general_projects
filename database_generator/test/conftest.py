import os
import shutil
import pytest

TEMP_PATH = "/tmp/test_outcomes"

def pytest_configure(config):
    """
    Pytest hook to configure the environment before tests are collected or run.
    Ensures required environment variables and directories exist.
    """
    os.environ["PATH_TO_SAVE_THE_OUTCOMES"] = TEMP_PATH
    # Ensure the directory exists
    os.makedirs(TEMP_PATH, exist_ok=True)

def pytest_unconfigure(config):
    """
    Pytest hook to clean up after the test session.
    Removes the temporary directory created for tests.
    """
    if os.path.exists(TEMP_PATH):
        shutil.rmtree(TEMP_PATH)

# The config argument in pytest_configure and pytest_unconfigure is passed automatically by Pytest.
# It provides access to the Pytest configuration object, 
# which contains metadata about the current test session, including command-line options,
# configuration files, and plugin states.