#!/bin/bash
# run.sh - Setup Python environment and run main application
# 
# This script:
# 1. Creates a Python virtual environment named 'aprilTags'
# 2. Activates the virtual environment
# 3. Updates pip to the latest version
# 4. Installs required dependencies from requirements.txt
# 5. Runs the main.py application

# Exit on error
set -e

echo "🔧 Setting up Python environment and running application..."

# Check if virtual environment exists
if [ ! -d "aprilTags" ]; then
    echo "Creating virtual environment 'aprilTags'..."
    python3 -m venv aprilTags
else
    echo "Using existing 'aprilTags' virtual environment..."
fi

# Activate virtual environment
echo "Activating virtual environment..."
source aprilTags/bin/activate

# Update pip
echo "Updating pip to latest version..."
pip install -U pip

# Install dependencies
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

# Run the application
echo "Running the application..."
python3 main.py