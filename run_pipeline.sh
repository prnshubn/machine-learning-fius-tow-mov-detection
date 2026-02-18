#!/bin/bash

# This script runs the full data processing and model training pipeline.

echo "--- Setting up Python virtual environment ---"

# Deactivate any active virtual environment
if command -v deactivate &> /dev/null; then
    deactivate
fi

# Remove the old venv if it exists
if [ -d "venv" ]; then
    echo "Removing old virtual environment..."
    rm -rf venv
fi

# Create a new virtual environment
echo "Creating new virtual environment..."
python3 -m venv venv
if [ $? -ne 0 ]; then
    echo "Error: Failed to create virtual environment. Aborting."
    exit 1
fi

# Activate the virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "Error: Failed to install dependencies. Aborting."
    exit 1
fi

echo "--- Virtual environment setup complete ---"


echo "--- Running Pipeline ---"

# --- 1. Build Feature Dataset ---
echo "--- Step 1: Building feature dataset ---"
python3 src/data/01_build_features.py
if [ $? -ne 0 ]; then
    echo "Error: Failed to build feature dataset. Aborting."
    exit 1
fi
echo "--- Feature dataset built successfully ---"

# --- 2. Refine Labels ---
echo "--- Step 2: Refining labels ---"
python3 src/data/02_refine_labels.py
if [ $? -ne 0 ]; then
    echo "Error: Failed to refine labels. Aborting."
    exit 1
fi
echo "--- Labels refined successfully ---"

# --- 3. Compare Classifiers and Train Best Model ---
echo "--- Step 3: Comparing classifiers and training the best model ---"
python3 src/models/03_train_and_compare.py
if [ $? -ne 0 ]; then
    echo "Error: Failed to train and compare models. Aborting."
    exit 1
fi
echo "--- Best model trained and saved successfully ---"

# --- 4. Make Prediction ---
echo "--- Step 4: Making a prediction on a sample file ---"
python3 src/models/04_predict.py data/raw/signal_1500_metal_plate.csv
if [ $? -ne 0 ]; then
    echo "Error: Failed to make prediction. Aborting."
    exit 1
fi
echo "--- Prediction made successfully ---"

# --- 5. Generate Visualizations ---
echo "--- Step 5: Generating visualizations ---"
python3 src/visualization/plot_label_distribution.py
python3 src/visualization/plot_feature_distribution.py
python3 src/visualization/plot_distance_over_time.py data/raw/signal_1500_metal_plate.csv
python3 src/visualization/plot_classifier_comparison.py
echo "--- Visualizations generated successfully ---"

echo "--- Pipeline Finished ---"
