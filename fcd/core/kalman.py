"""
FCD-PSE Kalman Filter Components
=================================
Bayesian state estimation and update rules for FCD transformation.
"""

import numpy as np
from typing import Tuple, Optional
from .primitives import MathPrimitives


class KalmanFilter:
    """Kalman filter implementation for FCD state updates."""
    
    def __init__(self, state_dim: int = 3, obs_noise_scale: float = 0.1):
        """
        Initialize Kalman filter.
        
        Parameters:
        -----------
        state_dim : int
            Dimension of state vector (default: 3 for Trend, Mom, Vol)
        obs_noise_scale : float
            Observation noise scaling factor
        """
        self.state_dim = state_dim
        self.obs_noise_scale = obs_noise_scale
        
        # Observation noise covariance (diagonal)
        self.obs_noise = np.eye(state_dim) * obs_noise_scale
    
    def predict(self, 
                state: np.ndarray, 
                state_cov: np.ndarray, 
                transition_matrix: Optional[np.ndarray] = None,
                process_noise: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Kalman prediction step.
        
        Parameters:
        -----------
        state : np.ndarray
            Current state estimate (shape: [state_dim])
        state_cov : np.ndarray
            Current state covariance (shape: [state_dim, state_dim])
        transition_matrix : np.ndarray, optional
            State transition matrix (default: identity)
        process_noise : np.ndarray, optional
            Process noise covariance (default: scaled identity)
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray] : (predicted_state, predicted_cov)
        """
        if transition_matrix is None:
            transition_matrix = np.eye(self.state_dim)
        
        if process_noise is None:
            # Default process noise based on state covariance
            process_noise = state_cov * 0.1
        
        # Predict state
        predicted_state = transition_matrix @ state
        
        # Predict covariance
        predicted_cov = transition_matrix @ state_cov @ transition_matrix.T + process_noise
        
        return predicted_state, predicted_cov
    
    def update(self, 
               predicted_state: np.ndarray, 
               predicted_cov: np.ndarray, 
               observation: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Kalman update step using observation.
        
        Parameters:
        -----------
        predicted_state : np.ndarray
            Predicted state from prediction step (shape: [state_dim])
        predicted_cov : np.ndarray
            Predicted covariance (shape: [state_dim, state_dim])
        observation : np.ndarray
            Observed state (shape: [state_dim])
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray, np.ndarray] : (updated_state, updated_cov, kalman_gain)
        """
        # Innovation (measurement residual)
        innovation = observation - predicted_state
        
        # Innovation covariance
        innovation_cov = predicted_cov + self.obs_noise
        
        # Kalman gain
        kalman_gain = predicted_cov @ MathPrimitives.safe_inverse(innovation_cov)
        
        # Update state
        updated_state = predicted_state + kalman_gain @ innovation
        
        # Update covariance
        identity = np.eye(self.state_dim)
        updated_cov = (identity - kalman_gain) @ predicted_cov
        
        return updated_state, updated_cov, kalman_gain
    
    def compute_gain(self, 
                     predicted_cov: np.ndarray, 
                     obs_noise: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute Kalman gain directly.
        
        Parameters:
        -----------
        predicted_cov : np.ndarray
            Predicted state covariance
        obs_noise : np.ndarray, optional
            Observation noise covariance (default: self.obs_noise)
            
        Returns:
        --------
        np.ndarray : Kalman gain matrix
        """
        if obs_noise is None:
            obs_noise = self.obs_noise
        
        innovation_cov = predicted_cov + obs_noise
        kalman_gain = predicted_cov @ MathPrimitives.safe_inverse(innovation_cov)
        
        return kalman_gain
    
    def adaptive_obs_noise(self, 
                          predicted_state: np.ndarray, 
                          observation: np.ndarray,
                          base_noise: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Adaptively adjust observation noise based on innovation magnitude.
        
        Parameters:
        -----------
        predicted_state : np.ndarray
            Predicted state
        observation : np.ndarray
            Observed state
        base_noise : np.ndarray, optional
            Base noise covariance
            
        Returns:
        --------
        np.ndarray : Adjusted observation noise covariance
        """
        if base_noise is None:
            base_noise = self.obs_noise
        
        # Calculate innovation magnitude
        innovation = observation - predicted_state
        innovation_mag = MathPrimitives.l2_norm(innovation)
        
        # Scale noise adaptively (larger innovation = larger noise)
        scale_factor = 1.0 + np.tanh(innovation_mag)
        
        return base_noise * scale_factor
