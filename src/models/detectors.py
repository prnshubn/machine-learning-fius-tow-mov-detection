"""
Custom ML Detectors: Specialized Algorithms for One-Class Classification.

This module contains specialized model architectures used for motion detection.
The primary detector is an Autoencoder, which is trained to reconstruct 'Towards' 
motion patterns. Any movement that cannot be reconstructed with low error 
(high reconstruction error) is flagged as an anomaly ('Not Towards').

Industry Approach:
- Reconstruction-Error Thresholding: Uses a quantile-based threshold (e.g., 95%) 
  to define the boundary of 'normal' (Towards) behavior.
- Scikit-Learn Compatibility: Inherits from BaseEstimator to allow seamless 
  integration into Pipelines and GridSearch.
"""

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.base import BaseEstimator

class AutoencoderDetector(BaseEstimator):
    """
    An MLP-based Autoencoder for Anomaly Detection.

    The model learns a compressed representation of the input features and 
    attempts to reconstruct them. Training is performed ONLY on 'Towards' data.

    Attributes:
        hidden_layer_sizes (tuple): Architecture of the encoder/decoder.
        threshold_quantile (float): The sensitivity of the anomaly detector.
        threshold_ (float): The derived reconstruction error cutoff.
    """

    def __init__(self, hidden_layer_sizes=(4, 2, 4), threshold_quantile=0.95):
        """
        Initializes the Autoencoder architecture.

        Only stores hyperparameters here — the MLPRegressor is intentionally
        NOT built in __init__. sklearn's BaseEstimator.set_params() updates
        instance attributes but cannot rebuild objects created at construction
        time, so GridSearch / Pipeline cloning would silently use the wrong
        architecture. Building the model inside fit() guarantees it always
        reflects the current values of hidden_layer_sizes and threshold_quantile.

        Args:
            hidden_layer_sizes (tuple): Hidden layer configuration.
            threshold_quantile (float): Quantile of training error used as threshold.
        """
        self.hidden_layer_sizes = hidden_layer_sizes
        self.threshold_quantile = threshold_quantile
        self.threshold_ = None

    def fit(self, X, y=None):
        """
        Trains the Autoencoder to reconstruct the input X.

        Args:
            X (np.ndarray): Training data (Target class only).
            y: Ignored (Compatibility with sklearn Pipeline).

        Returns:
            self
        """
        # Build the MLPRegressor here so that any set_params() call made
        # between construction and fitting is honoured correctly.
        self.model = MLPRegressor(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation='relu',
            solver='adam',
            max_iter=500,
            random_state=42,
        )
        # Neural net learns the identity function: f(X) -> X
        self.model.fit(X, X)
        
        # Calculate reconstruction errors on the training set
        X_pred = self.model.predict(X)
        errors = np.mean(np.square(X - X_pred), axis=1)
        
        # Set the anomaly threshold based on the distribution of training errors
        self.threshold_ = np.quantile(errors, self.threshold_quantile)
        return self

    def predict(self, X):
        """
        Classifies input samples based on reconstruction error.

        Args:
            X (np.ndarray): Data to classify.

        Returns:
            np.ndarray: 1 for Inliers (Towards), -1 for Outliers (Not Towards).
        """
        X_pred = self.model.predict(X)
        errors = np.mean(np.square(X - X_pred), axis=1)
        
        # Compare error to the learned threshold
        return np.where(errors <= self.threshold_, 1, -1)
