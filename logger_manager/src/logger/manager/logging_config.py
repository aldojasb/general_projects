import logging
import os


def setup_logging(log_file_path=None, log_level=logging.INFO):
    """
    Setup logging configuration.

    Args:
        log_file_path (str): Path to the log file. If None, logs will be displayed on the console.
        log_level (int): Logging level. Default is logging.INFO.
    """
    # Updated format to include the logger's name
    log_format = "[%(name)s]:%(lineno)s - %(asctime)s - %(levelname)s: %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    if log_file_path:
        # Ensure the log directory exists
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

        logging.basicConfig(
            level=log_level,
            format=log_format,
            datefmt=date_format,
            handlers=[logging.FileHandler(log_file_path), logging.StreamHandler()],
        )
    else:
        logging.basicConfig(level=log_level, format=log_format, datefmt=date_format)


def setup_logging_for_this_script(log_level=logging.INFO) -> None:
    """
    Set up logging for the current script, ensuring the log directory and file exist.

    Args:
        log_level (int): Logging level. Default is logging.INFO.
    """
    # Retrieve the base path for saving logs from the environment variable
    path_to_save_the_logs = os.getenv("PATH_TO_SAVE_THE_LOGS")

    # Check if the environment variable is defined
    if not path_to_save_the_logs:
        raise EnvironmentError(
            "Environment variable 'PATH_TO_SAVE_THE_LOGS' is not defined. "
            "Please set it in your environment or .env file."
        )

    # Validate that the path exists and is writable
    if not os.path.exists(path_to_save_the_logs):
        raise FileNotFoundError(
            f"The specified path '{path_to_save_the_logs}' does not exist. "
            "Please create the directory or specify a valid path."
        )
    if not os.access(path_to_save_the_logs, os.W_OK):
        raise PermissionError(
            f"The specified path '{path_to_save_the_logs}' is not writable. "
            "Please check the directory permissions."
        )

    # Setup logging directory and file
    log_dir_path = os.path.join(path_to_save_the_logs, "tmp")
    os.makedirs(log_dir_path, exist_ok=True)  # Ensure that the log directory exists
    log_file_path = os.path.join(log_dir_path, "logs.log")  # Create the log file path

    # Call the setup_logging function with the constructed log file path
    setup_logging(log_file_path=log_file_path, log_level=log_level)