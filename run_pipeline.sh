#!/bin/bash
# Master Pipeline for Acoustic Shield: Proactive Safety via Raw Ultrasonic Signal Analysis
# This script automates the entire lifecycle: Extraction -> Labeling -> Training -> Simulation -> Visualization.

LOG_FILE="processing.log"

run_pipeline() {
    echo "--- Project Run Started: $(date) ---"
    
    # --- Environment Setup ---
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
    
    # Clean PYTHONPATH
    export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"

    echo "Syncing dependencies..."
    pip install --upgrade pip &> /dev/null
    pip install -r requirements.txt &> /dev/null

    # --- Step 1: Feature Extraction ---
    echo "--- Step 1: Extracting Features (Pure Sensor Data) ---"
    python3 src/data/feature_extractor.py
    if [ $? -ne 0 ]; then echo "Error in Step 1: Feature Extraction"; return 1; fi

    # --- Step 2: Ground Truth Labeling ---
    echo "--- Step 2: Generating Labels & Kinematic TTC ---"
    python3 src/data/label_generator.py
    if [ $? -ne 0 ]; then echo "Error in Step 2: Label Generation"; return 1; fi

    # --- Step 3: Model Training (Master Pipeline) ---
    # This now handles both OCC Algorithm Sweep (Stage 1) and TTC Regression (Stage 2)
    echo "--- Step 3: Training Master ML Pipeline (OCC Sweep & TTC Improvement) ---"
    python3 src/models/model_trainer.py
    if [ $? -ne 0 ]; then echo "Error in Step 3: Model Training"; return 1; fi

    # --- Step 4: Real-time Simulation (All Files) ---
    echo "--- Step 4: Simulating Real-time Prediction on all Raw Data ---"
    RAW_FILES=$(ls data/raw/*.csv)
    for file in $RAW_FILES; do
        echo "Processing: $(basename "$file")"
        python3 src/models/predictor.py "$file"
        if [ $? -ne 0 ]; then echo "Warning: Error predicting on $file"; fi
    done

    # --- Step 5: Visualizations ---
    echo "--- Step 5: Generating Global Visualizations ---"
    python3 src/visualization/plot_label_distribution.py
    python3 src/visualization/plot_classifier_comparison.py
    
    echo "--- Generating Distance Plots for all Raw Data ---"
    for file in $RAW_FILES; do
        python3 src/visualization/plot_distance_over_time.py "$file"
    done

    echo "--- Pipeline Finished Successfully: $(date) ---"
}

# Execute and log
run_pipeline 2>&1 | tee "$LOG_FILE"
