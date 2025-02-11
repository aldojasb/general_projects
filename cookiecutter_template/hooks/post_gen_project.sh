#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status
set -x  # Print commands and their arguments as they are executed

# Define the path to the templates and the destination directory
TEMPLATES_PATH="/workspace/general_projects/cookiecutter_template/_templates"
DEST_DIR="src/{{ cookiecutter.project_name.lower().replace(' ', '/').replace('-', '/').replace('_', '/') }}"

# Debug output
echo "TEMPLATES_PATH is $TEMPLATES_PATH"
echo "DEST_DIR is $DEST_DIR"

# Create the src directory
mkdir -p "$DEST_DIR"
echo "The basic src directory was created successfully."

# Copy template Python files to the new src directory
cp "$TEMPLATES_PATH/__init__.py" "$DEST_DIR/"
cp "$TEMPLATES_PATH/get_data.py" "$DEST_DIR/"
cp "$TEMPLATES_PATH/helpers.py" "$DEST_DIR/"
cp "$TEMPLATES_PATH/main.py" "$DEST_DIR/"
cp "$TEMPLATES_PATH/predict.py" "$DEST_DIR/"
cp "$TEMPLATES_PATH/process_data.py" "$DEST_DIR/"
cp "$TEMPLATES_PATH/train.py" "$DEST_DIR/"

echo "The extra src files were created successfully."

# Clear Poetry cache to avoid conflicts
# It only removes the current project's virtual environment.
# It doesn’t touch other projects.

if poetry env info --path &>/dev/null; then
    poetry env remove $(poetry env info --path)
    echo "Poetry virtual environment for this project has been removed."
else
    echo "No existing virtual environment found for this project."
fi

