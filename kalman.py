"""A fixed-parameter linear Kalman-filter implementation for 2D tracking.

The state vector is ordered as ``[x, vx, y, vy]`` and the measurement vector
contains position ``[x, y]``. The transition matrix uses a time step of 0.05,
and the observation matrix selects the two position components. Process and
measurement uncertainty are fixed in the implementation rather than supplied
per update.
"""

import numpy as np

class KMfilter():
    """Linear Kalman filter with a fixed four-component motion state.

    The state is ``[x, vx, y, vy]`` and measurements contain ``[x, y]``.
    The transition matrix ``A``, observation matrix ``H``, nominal step, and
    measurement covariance ``c_w`` are initialized as instance attributes.
    The control matrix ``B`` is set to zero and is not used by the prediction
    methods.

    Note:
        Covariance prediction adds the scalar ``0.05`` to the matrix result.
        The covariance update follows the expression implemented in
        :meth:`cov_update`; it is not the Joseph stabilized form.

    Attributes:
        step (float): Time step used to construct the transition matrix.
        A (numpy.ndarray): Four-by-four state transition matrix.
        B (int): Zero-valued control matrix placeholder; not used by prediction.
        H (numpy.ndarray): Observation matrix selecting the x and y positions.
        c_w (numpy.ndarray): One-by-one measurement covariance array.
    """
    def __init__(self):
        """Initialize the transition, observation, and noise parameters.

        The time step is 0.05, the state transition assumes constant velocity,
        and ``c_w`` is initialized to a 1-by-1 array containing 0.5.

        Returns:
            None.
        """
        self.step = .05
        self.A = np.array([[1, self.step, 0, 0],[ 0, 1, 0, 0],[ 0, 0, 1, self.step],[ 0, 0 ,0, 1]])
        self.B = 0
        self.H = np.array([[1, 0, 0, 0],[ 0, 0, 1, 0]])
        self.c_w = np.array([0.5]).reshape(1,1)


    def state_predict(self, x_prior):
        """Predict the state using the configured transition matrix.

        Args:
            x_prior (numpy.ndarray): Previous state estimate, expected to
                have shape ``(4, 1)``.

        Returns:
            numpy.ndarray: Predicted state ``A @ x_prior``, with shape
            ``(4, 1)``.
        """
        x_pred = self.A @ x_prior

        return x_pred

    def cov_predict(self, cov_prior):
        """Predict covariance using the transition matrix and fixed offset.

        Args:
            cov_prior (numpy.ndarray): Previous state covariance, expected to
                have shape ``(4, 4)``.

        Returns:
            numpy.ndarray: ``A @ cov_prior @ A.T + 0.05``. NumPy broadcasts
            the scalar offset across the resulting matrix.
        """
        cov_pred = self.A @ cov_prior @ self.A.T + .05

        return cov_pred

    def kalman_gain(self, cov_prior):
        """Compute the gain from a predicted state covariance.

        Args:
            cov_prior (numpy.ndarray): Predicted covariance, expected to have
                shape ``(4, 4)``.

        Returns:
            numpy.ndarray: Gain matrix computed as
            ``P @ H.T @ inv(c_w + H @ P @ H.T)``, where ``P`` is
            ``cov_prior``. With the configured matrices, its shape is
            ``(4, 2)``.

        Raises:
            numpy.linalg.LinAlgError: If the innovation covariance matrix is
                singular.
        """
        k = cov_prior @ self.H.T @ (np.linalg.inv(self.c_w + self.H @ cov_prior @ self.H.T))

        return k

    def state_update(self, x_prior, k, measurement):
        """Correct a predicted state with a position measurement.

        Args:
            x_prior (numpy.ndarray): Predicted state, expected to have shape
                ``(4, 1)``.
            k (numpy.ndarray): Kalman gain, expected to have shape ``(4, 2)``.
            measurement (numpy.ndarray): Observed ``[x, y]`` position,
                expected to have shape ``(2, 1)``.

        Returns:
            numpy.ndarray: Corrected state computed from the measurement
            residual, with shape ``(4, 1)``.
        """
        x_update = x_prior + k @ ( measurement - self.H @ x_prior )

        return x_update

    def cov_update(self, cov_prior, k):
        """Update covariance using the implementation's correction expression.

        Args:
            cov_prior (numpy.ndarray): Predicted covariance, expected to have
                shape ``(4, 4)``.
            k (numpy.ndarray): Kalman gain, expected to have shape ``(4, 2)``.

        Returns:
            numpy.ndarray: Result of ``(1 - k) @ H @ cov_prior`` using
            NumPy's elementwise scalar subtraction and matrix multiplication.
            With the configured matrix dimensions, the returned array has
            shape ``(4, 4)``.
        """
        cov_update = (1 - k ) @ self.H @ cov_prior
        return cov_update
