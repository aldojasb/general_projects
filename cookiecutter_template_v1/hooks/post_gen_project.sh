#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status
set -x  # Print commands and their arguments as they are executed

# Define the path to the templates and the destination directory
TEMPLATES_PATH="/workspace/general_projects/cookiecutter_template_v1/_templates"
DEST_DIR="src/{{ cookiecutter.project_name.lower().replace(' ', '_').replace('-', '_') }}"

# Debug output
echo "TEMPLATES_PATH is $TEMPLATES_PATH"
echo "DEST_DIR is $DEST_DIR"

# Create the src directory
mkdir -p "$DEST_DIR"
echo "The basic src directory was created successfully."

# Copy template Python files to the new src directory
cp "$TEMPLATES_PATH/__init__.py" "$DEST_DIR/"
cp "$TEMPLATES_PATH/evaluate.py" "$DEST_DIR/"
cp "$TEMPLATES_PATH/get_data.py" "$DEST_DIR/"
cp "$TEMPLATES_PATH/helpers.py" "$DEST_DIR/"
cp "$TEMPLATES_PATH/logging_configuration.py" "$DEST_DIR/"
cp "$TEMPLATES_PATH/main.py" "$DEST_DIR/"
cp "$TEMPLATES_PATH/predict.py" "$DEST_DIR/"
cp "$TEMPLATES_PATH/process_data.py" "$DEST_DIR/"
cp "$TEMPLATES_PATH/train.py" "$DEST_DIR/"

echo "The extra src files were created successfully."

# Configure Poetry to create the virtual environment inside the project directory
echo "Configuring Poetry to create virtual environment inside the project directory"
poetry config virtualenvs.in-project true

# Install dependencies
echo "Installing dependencies with Poetry"
poetry install

# Output the configured virtual environment path
echo "Poetry is configured to use the following virtual environment:"
poetry env info --path

# Activate the virtual environment
VENV_PATH=$(poetry env info --path)
source "$VENV_PATH/bin/activate"

# Perform the editable installation of the project
echo "Installing the project in editable mode"
pip install -e .

# Create a new Jupyter kernel for the virtual environment using the project name
KERNEL_NAME="{{ cookiecutter.project_name.lower().replace(' ', '_').replace('-', '_') }}"
KERNEL_DISPLAY_NAME="{{ cookiecutter.project_name.lower().replace(' ', '_').replace('-', '_') }}"
echo "Creating a new Jupyter kernel for the virtual environment"
python -m ipykernel install --user --name="$KERNEL_NAME" --display-name "$KERNEL_DISPLAY_NAME"

echo "Setup complete. The Jupyter kernel '$KERNEL_DISPLAY_NAME' has been created."
