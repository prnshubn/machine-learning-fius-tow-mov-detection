#!/bin/bash
# =============================================================================
# Full Pipeline Runner: Ultrasonic Sensor TTC Prediction System
# Runs feature extraction → labeling → TTC regression model training.
# Real-time output is displayed in the terminal and saved to 'processing.log'.
# =============================================================================

LOG_FILE="processing.log"
SAMPLE_FILE="data/raw/signal_1500_metal_plate.csv"

# --- Colour helpers (silent if terminal doesn't support it) ---
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

print_step()  { echo -e "\n${CYAN}${BOLD}>>> $1${RESET}"; }
print_ok()    { echo -e "${GREEN}[OK]${RESET} $1"; }
print_warn()  { echo -e "${YELLOW}[WARN]${RESET} $1"; }
print_error() { echo -e "${RED}[ERROR]${RESET} $1"; }

# =============================================================================
run_pipeline() {
# =============================================================================

echo -e "\n${BOLD}============================================="
echo -e " Pipeline Started: $(date)"
echo -e "=============================================${RESET}"

# ---------------------------------------------------------------------------
# 0. Virtual environment setup
# ---------------------------------------------------------------------------
print_step "Step 0: Setting up Python virtual environment"

# Deactivate any currently active venv to avoid conflicts
if command -v deactivate &> /dev/null; then
    deactivate 2>/dev/null || true
fi

if [ ! -d "venv" ]; then
    echo "Creating new virtual environment..."
    python3 -m venv venv || { print_error "Failed to create venv. Is python3 installed?"; return 1; }
else
    echo "Using existing virtual environment."
fi

source venv/bin/activate || { print_error "Failed to activate venv."; return 1; }

# Ensure src/ is on the Python path so internal imports (src.data.*) resolve
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

echo "Syncing dependencies..."
pip install --upgrade pip -q
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt -q
    print_ok "Dependencies installed."
else
    print_warn "requirements.txt not found — skipping dependency install."
fi

# ---------------------------------------------------------------------------
# 1. Feature extraction
# ---------------------------------------------------------------------------
print_step "Step 1: Extracting features from raw sensor data"

if [ ! -d "data/raw" ] || [ -z "$(ls data/raw/*.csv 2>/dev/null)" ]; then
    print_error "No raw CSV files found in data/raw/. Cannot continue."
    return 1
fi

python3 src/data/feature_extractor.py
if [ $? -ne 0 ]; then
    print_error "Feature extraction failed. Check logs above."
    return 1
fi
print_ok "features.csv written to data/processed/"

# ---------------------------------------------------------------------------
# 2. Label generation + TTC computation
# ---------------------------------------------------------------------------
print_step "Step 2: Generating motion labels and computing TTC"

python3 src/data/label_generator.py
if [ $? -ne 0 ]; then
    print_error "Label generation failed. Check logs above."
    return 1
fi
print_ok "final_labeled_data.csv written to data/processed/"

# ---------------------------------------------------------------------------
# 3. TTC regression model training (Linear Regression + SVR)
# ---------------------------------------------------------------------------
print_step "Step 3: Training TTC regression models (Linear Regression + SVR)"

if [ ! -f "src/models/model_trainer.py" ]; then
    print_error "src/models/model_trainer.py not found."
    print_warn "Place the generated model_trainer.py at src/models/model_trainer.py and re-run."
    return 1
fi

python3 src/models/model_trainer.py
if [ $? -ne 0 ]; then
    print_error "Model training failed. Check logs above."
    return 1
fi
print_ok "Models saved to models/"

# ---------------------------------------------------------------------------
# 4. Real-time prediction demo (optional — runs if predictor exists)
# ---------------------------------------------------------------------------
print_step "Step 4: Real-time prediction demo"

if [ ! -f "src/models/predictor.py" ]; then
    print_warn "src/models/predictor.py not found — skipping demo. (Not yet implemented)"
elif [ ! -f "$SAMPLE_FILE" ]; then
    print_warn "Sample file '$SAMPLE_FILE' not found — skipping demo."
else
    python3 src/models/predictor.py "$SAMPLE_FILE"
    if [ $? -ne 0 ]; then
        print_warn "Prediction demo failed — pipeline will continue."
    else
        print_ok "Prediction demo completed."
    fi
fi

# ---------------------------------------------------------------------------
# 5. Visualizations (optional — each runs only if the script exists)
# ---------------------------------------------------------------------------
print_step "Step 5: Generating visualizations"

PLOTS_RUN=0

run_plot() {
    local script="$1"; shift
    if [ -f "$script" ]; then
        python3 "$script" "$@"
        if [ $? -eq 0 ]; then
            print_ok "$(basename $script) completed."
            PLOTS_RUN=$((PLOTS_RUN + 1))
        else
            print_warn "$(basename $script) failed — continuing."
        fi
    else
        print_warn "$script not found — skipping."
    fi
}

run_plot "src/visualization/plot_label_distribution.py"
run_plot "src/visualization/plot_distance_over_time.py" "$SAMPLE_FILE"
run_plot "src/visualization/plot_classifier_comparison.py"

if [ $PLOTS_RUN -eq 0 ]; then
    print_warn "No visualization scripts found. Add them to src/visualization/ to enable."
fi

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo -e "\n${BOLD}${GREEN}============================================="
echo -e " Pipeline Finished Successfully: $(date)"
echo -e "=============================================${RESET}"
echo ""
echo "  Output files:"
echo "    data/processed/features.csv"
echo "    data/processed/final_labeled_data.csv"
echo "    models/linear_regression_ttc.joblib"
echo "    models/svr_ttc.joblib"
echo ""

# =============================================================================
}
# =============================================================================

# Run the pipeline and tee output to both terminal and log file
run_pipeline 2>&1 | tee "$LOG_FILE"
