"""
Shared Detector Classes for One-Class Classification.
"""
import numpy as np
from sklearn.neural_network import MLPRegressor

class AutoencoderDetector:
    """
    Autoencoder implementation using MLPRegressor.
    Trained to reconstruct 'Towards' data; anomalies yield high error.
    """
    def __init__(self, hidden_layer_sizes=(4, 2, 4), threshold_quantile=0.95):
        self.model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, 
                                  activation='relu', solver='adam', 
                                  max_iter=500, random_state=42)
        self.threshold_quantile = threshold_quantile
        self.threshold = None

    def fit(self, X):
        # Neural net learns to map X to itself
        self.model.fit(X, X)
        # Set outlier threshold based on training reconstruction error
        X_pred = self.model.predict(X)
        errors = np.mean(np.square(X - X_pred), axis=1)
        self.threshold = np.quantile(errors, self.threshold_quantile)
        return self

    def predict(self, X):
        X_pred = self.model.predict(X)
        errors = np.mean(np.square(X - X_pred), axis=1)
        # Returns 1 for Inliers, -1 for Outliers
        return np.where(errors <= self.threshold, 1, -1)
