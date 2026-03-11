
import numpy as np

class KalmanFilter:
    """
    A simple 1D Kalman Filter for tracking distance, velocity, and acceleration.
    State vector x = [distance, velocity, acceleration]^T
    """
    def __init__(self, dt=0.045, process_noise=0.01, measurement_noise=10.0):
        """
        Initialize the Kalman Filter.
        
        Args:
            dt (float): Time step between measurements (seconds). 
            process_noise (float): Tuning parameter (Trust in model).
            measurement_noise (float): Tuning parameter (Trust in sensor).
        """
        self.dt = dt
        
        # State vector [dist, vel, accel]
        self.x = np.zeros((3, 1))
        
        # State transition matrix (Constant Acceleration Model)
        self.F = np.array([
            [1, dt, 0.5 * dt**2],
            [0, 1, dt],
            [0, 0, 1]
        ])
        
        # Measurement matrix (we only measure distance)
        self.H = np.array([[1, 0, 0]])
        
        # Process noise covariance
        self.Q = np.eye(3) * process_noise
        
        # Measurement noise covariance
        self.R = np.array([[measurement_noise]])
        
        # Error covariance matrix
        self.P = np.eye(3) * 100.0  # High initial uncertainty
        
        self.initialized = False

    def update(self, z):
        """
        Perform one Predict-Update cycle.
        
        Args:
            z (float): The measured distance.
            
        Returns:
            tuple: (filtered_distance, filtered_velocity, filtered_acceleration)
        """
        if not self.initialized:
            self.x = np.array([[z], [0], [0]])
            self.initialized = True
            return self.x[0,0], self.x[1,0], self.x[2,0]

        # 1. Predict
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

        # 2. Update (Correction)
        y = z - np.dot(self.H, self.x)  # Innovation
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R # Innovation covariance
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S)) # Kalman Gain
        
        self.x = self.x + np.dot(K, y)
        self.P = self.P - np.dot(np.dot(K, self.H), self.P)
        
        return self.x[0,0], self.x[1,0], self.x[2,0]

def apply_kalman_filter(distances, timestamps, process_noise=0.01, measurement_noise=10.0):
    """
    Apply Kalman Filter to a sequence of distance measurements.
    
    Args:
        distances (list/Series): Raw distance measurements.
        timestamps (list/Series): Timestamps in milliseconds.
        
    Returns:
        tuple: (filtered_dist, filtered_vel, filtered_accel) as numpy arrays.
    """
    if len(distances) < 2:
        return np.array(distances), np.zeros(len(distances)), np.zeros(len(distances))
        
    # Calculate average dt from timestamps (ms to s)
    dt_avg = np.mean(np.diff(timestamps)) / 1000.0
    if np.isnan(dt_avg) or dt_avg <= 0:
        dt_avg = 0.045 # Fallback
        
    kf = KalmanFilter(dt=dt_avg, process_noise=process_noise, measurement_noise=measurement_noise)
    
    f_dist, f_vel, f_accel = [], [], []
    
    for d in distances:
        d_val, v_val, a_val = kf.update(d)
        f_dist.append(d_val)
        f_vel.append(v_val)
        f_accel.append(a_val)
        
    return np.array(f_dist), np.array(f_vel), np.array(f_accel)
