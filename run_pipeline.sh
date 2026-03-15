#!/bin/bash
# Master Pipeline for Acoustic Shield: Proactive Safety via Raw Ultrasonic Signal Analysis
# This script automates the entire lifecycle: Extraction -> Labeling -> Training -> Simulation -> Visualization.

LOG_FILE="processing.log"

run_pipeline() {
    echo "--- Project Run Started: $(date) ---"

    # --- Step 0: Cleanup ---
    # Delete old artifacts to ensure a fresh run and no orphaned files.
    echo "--- Step 0: Cleaning up old artifacts ---"
    rm -rf models/*.joblib
    rm -rf reports/*.csv
    rm -rf reports/figures/*.png
    # Optional: rm -rf data/processed/*.csv  (to force re-extraction)
    echo "  Old models, CSV reports, and figures deleted."

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

    # --- Step 3: Model Training ---
    echo "--- Step 3: Training Master ML Pipeline (OCC Sweep & TTC Regression) ---"
    python3 src/models/model_trainer.py
    if [ $? -ne 0 ]; then echo "Error in Step 3: Model Training"; return 1; fi

    # --- Step 4: Real-time Simulation (All Files) ---
    echo "--- Step 4: Simulating Real-time Prediction on all Raw Data ---"
    RAW_FILES=$(ls data/raw/*.csv 2>/dev/null)
    if [ -z "$RAW_FILES" ]; then
        echo "Warning: No raw CSV files found in data/raw/"
    else
        for file in $RAW_FILES; do
            echo "  Processing: $(basename "$file")"
            python3 src/models/predictor.py "$file"
            if [ $? -ne 0 ]; then echo "  Warning: Error predicting on $file"; fi
        done
    fi

    # --- Step 5: Visualizations ---
    echo "--- Step 5: Generating All Visualizations ---"
    mkdir -p reports/figures

    # Existing plots
    python3 src/visualization/plot_label_distribution.py
    if [ $? -ne 0 ]; then echo "  Warning: label distribution plot failed"; fi

    python3 src/visualization/plot_classifier_comparison.py
    if [ $? -ne 0 ]; then echo "  Warning: classifier comparison plot failed"; fi

    # New plots
    echo "  Generating confusion matrix heatmap..."
    python3 src/visualization/plot_confusion_matrix.py
    if [ $? -ne 0 ]; then echo "  Warning: confusion matrix plot failed"; fi

    echo "  Generating TTC prediction scatter..."
    python3 src/visualization/plot_ttc_prediction.py
    if [ $? -ne 0 ]; then echo "  Warning: TTC prediction plot failed"; fi

    echo "  Generating echo profiles by object type..."
    python3 src/visualization/plot_echo_profiles.py
    if [ $? -ne 0 ]; then echo "  Warning: echo profiles plot failed"; fi

    echo "  Generating velocity timeline with motion labels..."
    python3 src/visualization/plot_velocity_timeline.py
    if [ $? -ne 0 ]; then echo "  Warning: velocity timeline plot failed"; fi

    echo "  Generating feature correlation heatmap..."
    python3 src/visualization/plot_feature_correlation.py
    if [ $? -ne 0 ]; then echo "  Warning: feature correlation plot failed"; fi

    # Distance plots for each raw file
    echo "--- Generating Distance Plots for all Raw Data ---"
    if [ -n "$RAW_FILES" ]; then
        for file in $RAW_FILES; do
            python3 src/visualization/plot_distance_over_time.py "$file"
            if [ $? -ne 0 ]; then echo "  Warning: distance plot failed for $file"; fi
        done
    fi

    echo ""
    echo "--- All outputs saved to reports/figures/ ---"
    echo "--- Pipeline Finished Successfully: $(date) ---"
}

# Execute and log
run_pipeline 2>&1 | tee "$LOG_FILE"
