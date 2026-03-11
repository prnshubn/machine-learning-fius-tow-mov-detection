# Acoustic Shield: Proactive Safety via Raw Ultrasonic Signal Analysis

## 🌟 Project Vision: From Sensing to Safety
Standard ultrasonic sensors are typically used for simple distance measurement. This project transforms that passive data into a **proactive safety system**. By bypassing pre-calculated distance metrics and analyzing **Raw ADC (Analog-to-Digital Converter) Reflections**, we identify complex movement patterns and predict Time-to-Collision (TTC) with high precision.

---

## 🚀 Key Technical Innovations

### 1. Pure Sensor Data (Beyond the Distance Column)
Traditional systems rely on a single distance value. Our pipeline implements **Robust First-Peak Detection**. We scan the raw ADC buffer to find the exact sample index where the first ultrasonic "Echo" occurs. This allows the model to learn the raw physical characteristics of the reflection rather than relying on a black-box distance calculation.

### 2. Quad-Scale Temporal Context (Dynamic Movement Recognition)
A single radar line is just a snapshot. Movement is a story told over time. We implement **Quad-Scale Windowing**:
*   **Micro Window (5 frames):** Reactive to high-speed bursts and immediate starts.
*   **Standard Window (10-25 frames):** Captures the core physical trend of a human approach.
*   **Macro Window (50 frames):** Identifies slow, steady creeps and filters out long-term environmental drift.
By calculating trends (deltas) and rolling averages across all four scales, the model can distinguish between a sensor glitch and real physical intent across a dynamic range of speeds.

### 3. Dual-Stage ML Architecture
We employ a sophisticated two-stage pipeline:
*   **Stage 1 (Classification):** Uses **One-Class Classification (OCC)** to determine if an object is "Towards" the sensor. We treat everything else (noise, stationary, receding) as an anomaly.
*   **Stage 2 (Regression):** If Stage 1 detects a threat, the system triggers a **Support Vector Regressor (SVR)** to predict the exact moment of impact (TTC).

---

## 🛠 Detailed Component Breakdown

### Step 1: Feature Extraction (`feature_extractor.py`)
*   **Echo Indexing:** Uses a 4x standard deviation threshold to find the first significant peak in the raw signal.
*   **Spectral Centroids:** Analyzes the frequency distribution of the reflection.
*   **Trend Vectors:** Calculates the "Echo Trend" across short and long time scales.

### Step 2: Ground Truth Labeling (`label_generator.py`)
*   Uses a **Kalman Filter** to derive hidden velocity from the noisy raw signal.
*   Applies **High-Sensitivity Thresholding (0.05)** to capture the precise beginning and end of every approach burst.
*   Trims leading/trailing "dead air" to balance the dataset for the training engine.

### Step 3: Model Training & Tuning (`model_trainer.py`)
*   **Algorithm Sweep:** Automatically trains and compares **One-Class SVM**, **Isolation Forest**, **Local Outlier Factor (LOF)**, and **Autoencoders**.
*   **Hyperparameter Tuning:** Iterates through different configurations (Narrow vs. Wide Autoencoders, varying SVM 'nu' values) to find the absolute peak F1-Score.

### Step 3b: TTC Regression (`ttc_trainer.py`)
*   Trains **Linear Regression** and **SVR** models specifically on approach sequences.
*   Uses kinematics and echo trends to map signal patterns to a countdown in seconds before impact.

### Step 4: Real-time Simulation (`predictor.py`)
*   Simulates a live deployment using a **Rolling Queue Buffer**.
*   Implements the "Two-Stage" logic: Only calculates TTC if the OCC model verifies an approach is in progress.

---

## 🏭 Industry Standards Followed
1.  **Modular Pipeline:** Clear separation of concerns (Extraction -> Labeling -> Training -> Inference).
2.  **Robust Logging:** Unified `processing.log` with real-time `tee` streaming for terminal visibility.
3.  **Reproducibility:** `run_pipeline.sh` automates the entire environment setup and execution.
4.  **Edge AI Optimization:** Prioritizes features that can be extracted directly from raw hardware buffers (ADC samples).

---

## 📊 How to Run
Execute the full automated pipeline:
```bash
bash run_pipeline.sh
```
All terminal output, including performance metrics and training progress, will be visible in the terminal and saved to **`processing.log`**. Visualizations are exported to **`reports/figures/`**.
