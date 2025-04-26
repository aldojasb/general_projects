# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.17.0
#   kernelspec:
#     display_name: logger_manager_test_1)
#     language: python
#     name: logger_manager_env
# ---

import logging
from logger.manager.logging_config import setup_logging_for_this_script
from dotenv import load_dotenv
import os

# +
# Load the .env file only if it exists
dotenv_path = '/workspace/general_projects/logger_manager/.env'
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)
    print(f"Loaded environment variables from {dotenv_path}")
else:
    print(f"No .env file found at {dotenv_path}, relying on system environment variables.")

# Access the environment variable, with a fallback
path_to_logs = os.getenv('PATH_TO_SAVE_THE_LOGS')
print(f"Logs will be saved to: {path_to_logs}")
# -

setup_logging_for_this_script()
# Get the logger for this module
logger = logging.getLogger(__name__)

logger.info("Initialize main.py")


