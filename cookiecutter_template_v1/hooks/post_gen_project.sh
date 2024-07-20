#!/bin/bash

# Define the path to the templates and the destination directory
TEMPLATES_PATH="/home/aldo/Repositories/general_projects/cookiecutter_template_v1/_templates"
DEST_DIR="src/{{ cookiecutter.project_name.lower().replace(' ', '_').replace('-', '_') }}"

# Create the src directory
mkdir -p "$DEST_DIR"
echo "The basic src directory was created successfully."

# Copy template Python files to the new src directory
cp "$TEMPLATES_PATH/evaluate.py" "$DEST_DIR/"
cp "$TEMPLATES_PATH/get_data.py" "$DEST_DIR/"
cp "$TEMPLATES_PATH/helpers.py" "$DEST_DIR/"
cp "$TEMPLATES_PATH/main.py" "$DEST_DIR/"
cp "$TEMPLATES_PATH/predict.py" "$DEST_DIR/"
cp "$TEMPLATES_PATH/process_data.py" "$DEST_DIR/"
cp "$TEMPLATES_PATH/train.py" "$DEST_DIR/"

echo "the extra src files were created successfully."
