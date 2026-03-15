"""
Kalman Filter module for tracking distance, velocity, and acceleration.

FIX (B) — Per-frame dt:
    KalmanFilter.update() now accepts an optional `dt` argument. When provided,
    the state-transition matrix F is recomputed on-the-fly with the real elapsed
    time between the current and previous sensor frames. This keeps kinematics
    perfectly synchronised with the real-world clock even when the sensor frame
    rate jitters. If dt is not supplied the filter falls back to the fixed dt
    set at construction time, preserving backwards compatibility with
    apply_kalman_filter() which already derives a mean dt from timestamps.
"""
import numpy as np


class KalmanFilter:
    """
    A 1-D Kalman Filter for tracking distance, velocity, and acceleration.
    State vector: x = [distance, velocity, acceleration]^T
    """

    def __init__(self, dt=0.045, process_noise=0.01, measurement_noise=10.0):
        """
        Args:
            dt (float): Default time step in seconds (used as fallback when
                        update() is called without an explicit dt).
            process_noise (float): Trust in the motion model (Q diagonal).
            measurement_noise (float): Trust in the sensor reading (R scalar).
        """
        self.dt = dt

        self.x = np.zeros((3, 1))
        self.F = self._make_F(dt)

        # Measurement matrix — we only observe distance
        self.H = np.array([[1, 0, 0]])

        self.Q = np.eye(3) * process_noise
        self.R = np.array([[measurement_noise]])
        self.P = np.eye(3) * 100.0  # High initial uncertainty

        self.initialized = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_F(dt):
        """Constant-acceleration state-transition matrix for a given dt."""
        return np.array([
            [1, dt, 0.5 * dt ** 2],
            [0,  1,            dt],
            [0,  0,             1],
        ])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, z, dt=None):
        """
        Perform one Predict-Update cycle.

        FIX (B): if `dt` is supplied and positive, F is rebuilt for this
        specific inter-frame interval before the prediction step. This
        eliminates tracking lag when sensor timing is irregular.

        Args:
            z  (float): Measured distance.
            dt (float | None): Actual elapsed time since the last frame (seconds).
                               If None, the constructor's default dt is used.

        Returns:
            tuple: (filtered_distance, filtered_velocity, filtered_acceleration)
        """
        if not self.initialized:
            self.x = np.array([[z], [0.0], [0.0]])
            self.initialized = True
            return self.x[0, 0], self.x[1, 0], self.x[2, 0]

        # --- Per-frame dt: rebuild F when a real interval is available ---
        if dt is not None and dt > 0:
            self.F = self._make_F(dt)

        # 1. Predict
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

        # 2. Update (Correction)
        y = z - np.dot(self.H, self.x)                          # Innovation
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R  # Innovation covariance
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S)) # Kalman gain

        self.x = self.x + np.dot(K, y)
        self.P = self.P - np.dot(np.dot(K, self.H), self.P)

        return self.x[0, 0], self.x[1, 0], self.x[2, 0]


def apply_kalman_filter(distances, timestamps, process_noise=0.01, measurement_noise=10.0):
    """
    Apply the Kalman Filter to a sequence of distance measurements.

    Derives a mean dt from the timestamp array (milliseconds) and passes it to
    the KalmanFilter constructor. Individual per-frame dts could also be passed
    via update(dt=...) here; using the mean is sufficient for batch processing
    because timestamp jitter averages out across a full session.

    Args:
        distances  (list | Series): Raw distance measurements.
        timestamps (list | Series): Timestamps in milliseconds.

    Returns:
        tuple: (filtered_dist, filtered_vel, filtered_accel) as numpy arrays.
    """
    if len(distances) < 2:
        return np.array(distances), np.zeros(len(distances)), np.zeros(len(distances))

    dt_avg = np.mean(np.diff(timestamps)) / 1000.0
    if np.isnan(dt_avg) or dt_avg <= 0:
        dt_avg = 0.045

    kf = KalmanFilter(dt=dt_avg, process_noise=process_noise,
                      measurement_noise=measurement_noise)

    f_dist, f_vel, f_accel = [], [], []
    for d in distances:
        d_val, v_val, a_val = kf.update(d)
        f_dist.append(d_val)
        f_vel.append(v_val)
        f_accel.append(a_val)

    return np.array(f_dist), np.array(f_vel), np.array(f_accel)
