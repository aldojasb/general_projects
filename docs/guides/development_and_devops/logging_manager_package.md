# 📖 `logging_manager` Package Documentation

------

## Overview

[logging_manager](https://github.com/aldojasb/general_projects/tree/main/logger_manager) is a lightweight, reusable Python package designed to **standardize and simplify logging setup** across multiple projects.
 It provides a consistent way to initialize loggers with both console and file handlers, ensuring that every script or application follows a **professional logging strategy** with minimal effort.

Instead of configuring logging manually in every project, developers can simply import and use `logging_manager` to instantly enable robust logging behaviors.

------

## Core Functions

### `setup_logging(log_file_path=None, log_level=logging.INFO)`

Sets up logging configuration with customizable options.

- **Parameters**:

  - `log_file_path` (str, optional): Path to the log file. If `None`, logs will be shown only in the console.
  - `log_level` (int, optional): Standard Python logging level (e.g., `logging.INFO`, `logging.DEBUG`).

- **Behavior**:

  - If a file path is provided, logs will be saved both to the file and printed to the console.

  - If no file path is provided, logs are only printed to the console.

  - Logs follow a standardized format, including:

    ```markdown
    [LoggerName]:LineNumber - Timestamp - LogLevel: Message
    ```

------

### `setup_logging_for_this_script(log_level=logging.INFO)`

Specialized helper function to **automatically configure logging** based on an environment variable (`PATH_TO_SAVE_THE_LOGS`).

- **Parameters**:
  - `log_level` (int, optional): Logging verbosity level (default: `logging.INFO`).
- **Behavior**:
  - Retrieves the base directory for saving logs from the environment variable `PATH_TO_SAVE_THE_LOGS`.
  - Validates that the path exists and is writable.
  - Creates a default subdirectory (`tmp/`) if necessary.
  - Sets up logging to a `logs.log` file inside the prepared directory.
  - Also streams logs to the console simultaneously.
- **Exceptions**:
  - Raises a clear `EnvironmentError` if the environment variable is missing.
  - Raises a `FileNotFoundError` if the directory does not exist.
  - Raises a `PermissionError` if the directory is not writable.

------

## Why Use a Centralized Logging Manager?

- ✅ **Consistency**: Avoid reinventing logging configuration for each project or script.
- ✅ **Portability**: Projects can move across machines/environments while maintaining clean, reliable logging.
- ✅ **Separation of concerns**: Application logic stays clean, with logging concerns handled by a dedicated, reusable component.
- ✅ **Error Handling**: Clear errors when the environment is misconfigured, helping catch deployment issues early.
- ✅ **Scalability**: Easy to extend the logging configuration later (e.g., adding rotating file handlers, cloud log streams).

------

## Best Practices Followed

- ✔️ Logging configuration is **isolated** from application code.
- ✔️ **Environment-driven configuration** improves portability across dev/staging/production.
- ✔️ **Graceful error handling** when environment setup is incorrect.
- ✔️ **Flexible logging outputs**: console-only for local dev, file+console for production.
- ✔️ **Standardized log format** for easier parsing and debugging.

------

## Example Usage

```python
from logging.manager.logging_config import setup_logging_for_this_script
import logging

# Initialize logging
setup_logging_for_this_script(log_level=logging.DEBUG)

# Use logger
logger = logging.getLogger(__name__)

logger.info("Application started.")
logger.debug("This is a debug message.")
logger.warning("This is a warning message.")
```

------

## Requirements

- Python 3.11+
- Environment variable `PATH_TO_SAVE_THE_LOGS` must be defined if using `setup_logging_for_this_script`.
