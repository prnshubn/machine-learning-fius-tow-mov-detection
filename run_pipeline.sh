#!/bin/bash

# This script runs the full data processing and model training pipeline.
# Real-time output is displayed in the terminal and saved to 'processing.log'.

LOG_FILE="processing.log"

# Function to run the pipeline logic
run_pipeline() {
    echo "--- Project Run Started: $(date) ---"

    echo "--- Setting up Python virtual environment ---"
    if command -v deactivate &> /dev/null; then
        deactivate
    fi

    if [ ! -d "venv" ]; then
        echo "Creating new virtual environment..."
        python3 -m venv venv
    else
        echo "Using existing virtual environment."
    fi

    source venv/bin/activate
    export PYTHONPATH=$PYTHONPATH:$(pwd)

    echo "Syncing dependencies..."
    pip install --upgrade pip &> /dev/null
    pip install -r requirements.txt &> /dev/null

    echo "--- Step 1: Extracting Features (Pure Sensor Data) ---"
    python3 src/data/feature_extractor.py
    if [ $? -ne 0 ]; then echo "Error in Step 1: Feature Extraction"; return 1; fi

    echo "--- Step 2: Generating Labels (Towards vs Not Towards) ---"
    python3 src/data/label_generator.py
    if [ $? -ne 0 ]; then echo "Error in Step 2: Label Generation"; return 1; fi

    echo "--- Step 3: Training & Comparing OCC Algorithms (SVM, Isolation Forest, LOF, Autoencoder) ---"
    python3 src/models/model_trainer.py
    if [ $? -ne 0 ]; then echo "Error in Step 3: Model Training"; return 1; fi

    echo "--- Step 3b: Training TTC Regression Models (Linear Regression, SVR) ---"
    python3 src/models/ttc_trainer.py
    if [ $? -ne 0 ]; then echo "Error in Step 3b: TTC Training"; return 1; fi

    echo "--- Step 4: Simulating Real-time Prediction ---"
    # Use a sample file for demonstration
    SAMPLE_FILE="data/raw/signal_1500_metal_plate.csv"
    if [ -f "$SAMPLE_FILE" ]; then
        python3 src/models/predictor.py "$SAMPLE_FILE"
        if [ $? -ne 0 ]; then echo "Error in Step 4: Prediction"; return 1; fi
    else
        echo "Warning: Sample file '$SAMPLE_FILE' not found. Skipping Step 4."
    fi

    echo "--- Step 5: Generating Visualizations ---"
    python3 src/visualization/plot_label_distribution.py
    python3 src/visualization/plot_distance_over_time.py "$SAMPLE_FILE"
    python3 src/visualization/plot_classifier_comparison.py

    echo "--- Pipeline Finished Successfully: $(date) ---"
}

# Execute the pipeline function and pipe output to tee
# This ensures the user sees logs in real-time while they are being saved
run_pipeline 2>&1 | tee "$LOG_FILE"
