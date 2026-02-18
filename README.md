# Ultrasonic Motion Detection using Machine Learning

This project provides a comprehensive pipeline to determine the direction of an object's motion relative to an ultrasonic sensor. By analyzing the raw signal data from the sensor, we use machine learning to classify the object's movement as 'approaching', 'receding', or 'stationary'.

## Core Concepts: The Physics and the Math

### Ultrasonic Sensing
An ultrasonic sensor measures distance by emitting a high-frequency sound pulse (a "chirp") and timing how long it takes for the echo to return. The sensor used in this project emits a 40 kHz pulse, which is above the range of human hearing. The time-of-flight of the echo is directly proportional to the distance of the object.

### The Doppler Effect: The Key to Detecting Motion
The Doppler effect is a fundamental principle in physics that describes the change in frequency of a wave in relation to an observer who is moving relative to the wave source. A common example is the change in pitch of a siren as it moves towards or away from you.

-   **Towards:** As a sound source moves towards you, the sound waves are compressed, leading to a higher frequency (higher pitch).
-   **Away:** As it moves away, the waves are stretched, resulting in a lower frequency (lower pitch).

This same principle applies to the ultrasonic sensor's echo. The sensor is stationary, but the object it's detecting is moving.

-   If an object is moving **towards** the sensor, the reflected sound waves are compressed, and the echo returns with a slightly **higher frequency** than the emitted 40 kHz.
-   If the object is moving **away** from the sensor, the waves are stretched, and the echo has a slightly **lower frequency**.

By detecting this frequency shift in the echo, we can determine the direction of the object's motion.

### From Time to Frequency: The Fast Fourier Transform (FFT)
The raw sensor data is a time-domain signal, which is a series of amplitude values over time. The FFT is a mathematical algorithm that transforms this signal into the frequency domain, showing us which frequencies are present in the signal and at what intensity.

By analyzing the frequency spectrum, we can extract features that capture the information about the Doppler shift. The key features extracted in this project are:
-   **Peak Frequency:** The frequency with the highest magnitude in the spectrum. This is the most direct indicator of the Doppler shift.
-   **Mean Frequency:** The average frequency of the spectrum, weighted by the magnitude of each frequency.
-   **Spectral Centroid:** The "center of mass" of the spectrum. It's another measure of the central tendency of the spectral energy.
-   **Spectral Skewness and Kurtosis:** These are statistical measures that describe the shape of the spectrum. They can provide additional information about the nature of the reflected signal.

## The Data: From Raw Signals to Labeled Features

### Raw Sensor Data (`data/raw/`)
The raw data is stored in CSV files in the `data/raw/` directory. Each file represents a continuous recording session. Based on the processing scripts, we can infer the structure of each row in these files:
-   **Columns 0-9:** Metadata about the measurement (e.g., timestamps, sensor configuration).
-   **Column 10:** The distance to the object, as calculated by the sensor's internal processing.
-   **Columns 11-16:** Additional metadata.
-   **Columns 17 onwards:** The raw, digitized amplitude readings from the sensor's Analog-to-Digital Converter (ADC), representing the reflected sound wave (the echo).

### Processed Data
-   `data/processed/features.csv`: This file stores the features extracted from the raw data. Each row corresponds to a single measurement and contains the extracted spectral features, the distance, and the original file label.
-   `data/processed/final_labeled_data.csv`: This is the final dataset used for training. It's the same as `features.csv` but with the motion labels ('approaching', 'receding', 'stationary') added.

## The Data Science Pipeline: From Raw Signal to Prediction

### Step 1: Feature Extraction (`src/data/01_build_features.py`)
-   **Goal:** To convert the raw, time-domain ADC signal into a set of numerical features that can be used by a machine learning model.
-   **Process:**
    1.  The script reads each raw CSV file from the `data/raw/` directory.
    2.  For each row (a single measurement), it isolates the ADC readings.
    3.  It applies the **Fast Fourier Transform (FFT)** to the ADC data. The FFT is a crucial step that decomposes the time-based signal into its constituent frequencies. This allows us to see the frequency spectrum of the echo and look for the Doppler shift.
    4.  From the spectrum, it extracts a set of spectral features.
-   **Output:** A single `features.csv` file containing the extracted features for all measurements.

### Step 2: Label Generation (`src/data/02_refine_labels.py`)
-   **Goal:** To create the "ground truth" labels that our model will learn to predict.
-   **Process:**
    1.  The script reads the `features.csv` file.
    2.  It groups the data by the original source file, treating each as a separate session.
    3.  Within each session, it calculates the difference in the `distance` column between each row and the one before it.
    4.  Based on this difference, it assigns a motion label.
-   **Output:** The `final_labeled_data.csv` file, which is ready for model training.

### Step 3: Model Training and Selection (`src/models/03_train_and_compare.py`)
-   **Goal:** To compare several machine learning models, select the best one based on performance, and save it for future use.
-   **Process:**
    1.  It loads the `final_labeled_data.csv` file.
    2.  The data is split into a training set (80%) and a testing set (20%).
    3.  A suite of classifiers (Random Forest, Logistic Regression, SVM, K-Nearest Neighbors, and Gradient Boosting) are trained and evaluated on the same data.
    4.  The script identifies the classifier with the highest accuracy.
    5.  The best-performing model is saved to `models/motion_detection_model.joblib`.
    6.  The performance metrics for all classifiers are saved to `reports/classifier_comparison_results.csv`.

### Step 4: Prediction (`src/models/04_predict.py`)
-   **Goal:** To use the trained model to make a prediction on a new, unseen data sample.
-   **Process:**
    1.  It loads the saved `motion_detection_model.joblib`.
    2.  It applies the same feature extraction process to a new data sample.
    3.  It feeds the extracted features to the loaded model to get a prediction.

## Classifier Selection Rationale

To ensure we selected a robust model, we compared the performance of several well-known classifiers on our dataset. The results are summarized below:

| Classifier | Accuracy | F1-Score (approaching) | F1-Score (receding) |
| :--- | :--- | :--- | :--- |
| **Gradient Boosting** | **0.8089** | **0.86** | **0.71** |
| **Random Forest** | 0.8017 | 0.85 | 0.70 |
| **Logistic Regression** | 0.7767 | 0.83 | 0.68 |
| **Support Vector Machine**| 0.7509 | 0.80 | 0.68 |
| **K-Nearest Neighbors** | 0.7137 | 0.78 | 0.57 |

Based on this comparison, **Gradient Boosting** was automatically selected as the best-performing model and saved for use in the prediction script. While Random Forest also performed very well, Gradient Boosting demonstrated a slight edge in overall accuracy and F1-Score for this specific dataset.

## Project Structure

The project is organized into a standard machine learning project structure to ensure clarity and maintainability:

-   `data/`: Contains the raw sensor data and the processed feature datasets.
-   `docs/`: For project documentation.
-   `models/`: Stores the final, trained machine learning model files.
-   `reports/`: Contains generated reports and figures.
-   `src/`: Contains all the source code for the project.
-   `run_pipeline.sh`: A shell script to run the entire pipeline from start to finish.

## How to Run the Project

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the Pipeline:**
    To run the entire pipeline, from data processing to prediction and visualization, simply execute the `run_pipeline.sh` script:
    ```bash
    bash run_pipeline.sh
    ```

## Visualizations and Interpretation

The pipeline generates several visualizations, which are saved in the `reports/figures/` directory.

-   **`confusion_matrix.png`:** Shows how the trained model performed on the test data.
-   **`label_distribution.png`:** Shows the distribution of the 'approaching', 'receding', and 'stationary' labels in the training data.
-   **`feature_distribution.png`:** Shows the distribution of the 'Peak Frequency' for each of the original file labels.
-   **`distance_over_time_signal_1500_metal_plate.png`:** A plot of the distance over time for one of the raw data files.
-   **`classifier_comparison.png`:** A bar chart visually comparing the performance of the different classifiers.
