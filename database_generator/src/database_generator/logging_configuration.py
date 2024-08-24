import logging
import os

def setup_logging(log_file_path=None, log_level=logging.INFO):
    """
    Setup logging configuration.
    
    Args:
        log_file_path (str): Path to the log file. If None, logs will be displayed on the console.
        log_level (int): Logging level. Default is logging.INFO.
    """
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    
    if log_file_path:
        # Ensure the log directory exists
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        
        logging.basicConfig(
            level=log_level,
            format=log_format,
            datefmt=date_format,
            handlers=[
                logging.FileHandler(log_file_path),
                logging.StreamHandler()
            ]
        )
    else:
        logging.basicConfig(
            level=log_level,
            format=log_format,
            datefmt=date_format
        )