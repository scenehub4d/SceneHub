#!/bin/bash

# Name of the Conda environment
ENV_NAME=3d

# Create the Conda environment with Python 3.9
echo "Creating Conda environment: $ENV_NAME"
conda create -n $ENV_NAME python=3.9 -y

# Activate the Conda environment
echo "Activating Conda environment: $ENV_NAME"
source ~/anaconda3/etc/profile.d/conda.sh
conda activate $ENV_NAME

# Install required Python packages
echo "Installing required Python packages..."
pip install opencv-python==4.11.0.86
pip install utils3d
# pip install git+https://github.com/microsoft/MoGe.git
pip install pykinect_azure
pip install open3d==0.19.0

# Confirm successful setup
echo "Conda environment '$ENV_NAME' setup complete."
echo "To activate it, run: conda activate $ENV_NAME"