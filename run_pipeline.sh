#!/bin/bash

# This script runs the full data processing and model training pipeline.

echo "--- Setting up Python virtual environment ---"

# Deactivate any active virtual environment
if command -v deactivate &> /dev/null; then
    deactivate
fi


# Create the virtual environment if it does not exist
if [ ! -d "venv" ]; then
    echo "Creating new virtual environment..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "Error: Failed to create virtual environment. Aborting."
        exit 1
    fi
else
    echo "Using existing virtual environment."
fi

# Activate the virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Add current directory to PYTHONPATH to allow module imports
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Install or upgrade dependencies as needed
echo "Checking and installing/upgrading dependencies from requirements.txt..."
pip install --upgrade pip
pip install --upgrade -r requirements.txt
if [ $? -ne 0 ]; then
    echo "Error: Failed to install or upgrade dependencies. Aborting."
    exit 1
fi

echo "--- Virtual environment setup complete ---"


echo "--- Running Pipeline ---"

# --- 1. Build Feature Dataset ---
echo "--- Step 1: Building feature dataset ---"
python3 -m src.data.01_build_features
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

# --- 3b. Train TTC Model ---
echo "--- Step 3b: Training TTC Model ---"
python3 src/models/05_train_ttc_model.py
if [ $? -ne 0 ]; then
    echo "Error: Failed to train TTC model. Aborting."
    exit 1
fi
echo "--- TTC model trained successfully ---"

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
python3 src/visualization/plot_feature_importance.py
python3 src/visualization/plot_distance_over_time.py data/raw/signal_1500_metal_plate.csv
python3 src/visualization/plot_classifier_comparison.py
echo "--- Visualizations generated successfully ---"

# --- 6. LLM Safety Demo ---
echo "--- Step 6: Running LLM Safety Module Demo ---"
python3 src/models/llm_safety_module.py
echo "--- LLM Safety Demo complete ---"

echo "--- Pipeline Finished ---"
