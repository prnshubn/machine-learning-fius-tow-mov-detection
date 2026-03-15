"""
Kinematic Tracking Module: Implements a 1D Kalman Filter for sensor fusion.

This module provides a Kalman Filter designed to estimate and smooth the distance,
velocity, and acceleration of objects from raw ultrasonic Time-of-Flight (ToF) data.
By modeling the system with a Constant Acceleration (CA) model, it reduces sensor
noise and provides reliable derivatives (velocity/acceleration) even with jittery data.

Industry Approach: 
- Uses a Discrete-Time State-Space representation.
- Employs a Predict-Update cycle to handle measurements.
- Dynamically recomputes the State Transition Matrix (F) to handle sampling jitter.
"""

import numpy as np


class KalmanFilter:
    """
    A 1D Kalman Filter for tracking distance, velocity, and acceleration.

    State Vector (x): [distance, velocity, acceleration]^T
    Model: Constant Acceleration (CA)

    Attributes:
        dt (float): Time step in seconds between predictions.
        x (np.ndarray): State estimate vector (3x1).
        P (np.ndarray): Estimate covariance matrix (3x3), representing uncertainty.
        F (np.ndarray): State transition matrix (3x3).
        H (np.ndarray): Measurement matrix (1x3), mapping state to measurement.
        Q (np.ndarray): Process noise covariance (3x3), represents model uncertainty.
        R (np.ndarray): Measurement noise covariance (1x1), represents sensor noise.
    """

    def __init__(self, dt=0.045, process_noise=0.01, measurement_noise=10.0):
        """
        Initializes the Kalman Filter with default or provided parameters.

        Args:
            dt (float): Default time step in seconds.
            process_noise (float): Tuning parameter for the model's 'trust' (Q).
            measurement_noise (float): Tuning parameter for the sensor's 'trust' (R).
        """
        self.dt = dt

        # Initialize state [distance; velocity; acceleration]
        self.x = np.zeros((3, 1))

        # Initial state transition matrix (Constant Acceleration Model)
        self.F = self._make_F(dt)

        # Measurement matrix: we only measure distance (z = [1 0 0] * x)
        self.H = np.array([[1, 0, 0]])

        # Process Noise Matrix (Q): represents how much the physical model can deviate
        self.Q = np.eye(3) * process_noise

        # Measurement Noise (R): represents the variance of the sensor noise
        self.R = np.array([[measurement_noise]])

        # Covariance Matrix (P): initial uncertainty (high values = low confidence)
        self.P = np.eye(3) * 100.0

        self.initialized = False

    @staticmethod
    def _make_F(dt):
        """
        Computes the State Transition Matrix (F) for a Constant Acceleration model.

        F = [ 1   dt   0.5*dt^2 ]  (New Distance)
            [ 0   1    dt       ]  (New Velocity)
            [ 0   0    1        ]  (New Acceleration)
        """
        return np.array([
            [1, dt, 0.5 * dt**2],
            [0, 1, dt],
            [0, 0, 1]
        ])

    def update(self, z, dt=None):
        """
        Performs one full Kalman Predict-Update cycle.

        Args:
            z (float): The current distance measurement from the sensor.
            dt (float, optional): The actual elapsed time since the last update.
                                  Used to handle variable sampling rates.

        Returns:
            tuple: (filtered_distance, filtered_velocity, filtered_acceleration)
        """
        # Handle first measurement: set initial state to measurement
        if not self.initialized:
            self.x = np.array([[z], [0.0], [0.0]])
            self.initialized = True
            return self.x[0, 0], self.x[1, 0], self.x[2, 0]

        # Update transition matrix if a custom dt is provided (jitter compensation)
        if dt is not None and dt > 0:
            self.F = self._make_F(dt)

        # ---------------------------------------------------------
        # 1. PREDICT: Project the state forward in time
        # ---------------------------------------------------------
        # x = F * x
        self.x = np.dot(self.F, self.x)
        # P = F * P * F^T + Q
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

        # ---------------------------------------------------------
        # 2. UPDATE: Correct the prediction with the new measurement
        # ---------------------------------------------------------
        # y = z - H * x (Innovation / Residual)
        y = z - np.dot(self.H, self.x)
        
        # S = H * P * H^T + R (Innovation Covariance)
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        
        # K = P * H^T * S^-1 (Optimal Kalman Gain)
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

        # x = x + K * y (Updated state estimate)
        self.x = self.x + np.dot(K, y)
        
        # P = (I - K * H) * P (Updated estimate covariance)
        self.P = self.P - np.dot(np.dot(K, self.H), self.P)

        return self.x[0, 0], self.x[1, 0], self.x[2, 0]


def apply_kalman_filter(distances, timestamps, process_noise=0.01, measurement_noise=10.0):
    """
    Batch wrapper to apply the Kalman Filter to a full session array.

    Args:
        distances (np.ndarray): Array of raw distance measurements.
        timestamps (np.ndarray): Array of timestamps in milliseconds.
        process_noise (float): Model uncertainty tuning.
        measurement_noise (float): Sensor noise tuning.

    Returns:
        tuple: (filtered_dist, filtered_vel, filtered_accel) as numpy arrays.
    """
    if len(distances) < 2:
        return np.array(distances), np.zeros(len(distances)), np.zeros(len(distances))

    # Calculate average time step for initialization
    dt_avg = np.mean(np.diff(timestamps)) / 1000.0
    if np.isnan(dt_avg) or dt_avg <= 0:
        dt_avg = 0.045

    kf = KalmanFilter(dt=dt_avg, process_noise=process_noise, measurement_noise=measurement_noise)

    f_dist, f_vel, f_accel = [], [], []
    for d in distances:
        d_val, v_val, a_val = kf.update(d)
        f_dist.append(d_val)
        f_vel.append(v_val)
        f_accel.append(a_val)

    return np.array(f_dist), np.array(f_vel), np.array(f_accel)
