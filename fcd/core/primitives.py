"""
FCD-PSE Mathematical Primitives
================================
Core statistical and mathematical functions required for FCD transformation.
"""

import numpy as np
from typing import Tuple, Optional
from scipy import linalg


class MathPrimitives:
    """Core mathematical operations for FCD-PSE indicator."""
    
    @staticmethod
    def ema(data: np.ndarray, length: int) -> np.ndarray:
        """
        Exponential Moving Average.
        
        Parameters:
        -----------
        data : np.ndarray
            Input price series
        length : int
            EMA period
            
        Returns:
        --------
        np.ndarray : EMA values
        """
        alpha = 2.0 / (length + 1)
        ema_values = np.zeros_like(data)
        ema_values[0] = data[0]
        
        for i in range(1, len(data)):
            ema_values[i] = alpha * data[i] + (1 - alpha) * ema_values[i-1]
            
        return ema_values
    
    @staticmethod
    def macd(data: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        MACD (Moving Average Convergence Divergence).
        
        Parameters:
        -----------
        data : np.ndarray
            Input price series
        fast : int
            Fast EMA period
        slow : int
            Slow EMA period
        signal : int
            Signal line period
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray, np.ndarray] : (MACD line, Signal line, Histogram)
        """
        ema_fast = MathPrimitives.ema(data, fast)
        ema_slow = MathPrimitives.ema(data, slow)
        macd_line = ema_fast - ema_slow
        signal_line = MathPrimitives.ema(macd_line, signal)
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    @staticmethod
    def true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
        """
        Calculate True Range for ATR.
        
        Parameters:
        -----------
        high : np.ndarray
            High prices
        low : np.ndarray
            Low prices
        close : np.ndarray
            Close prices
            
        Returns:
        --------
        np.ndarray : True Range values
        """
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First value uses only high-low
        
        return tr
    
    @staticmethod
    def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, length: int = 14) -> np.ndarray:
        """
        Average True Range.
        
        Parameters:
        -----------
        high : np.ndarray
            High prices
        low : np.ndarray
            Low prices
        close : np.ndarray
            Close prices
        length : int
            ATR period
            
        Returns:
        --------
        np.ndarray : ATR values
        """
        tr = MathPrimitives.true_range(high, low, close)
        atr_values = MathPrimitives.ema(tr, length)
        return atr_values
    
    @staticmethod
    def rolling_std(data: np.ndarray, length: int) -> np.ndarray:
        """
        Rolling standard deviation (volatility).
        
        Parameters:
        -----------
        data : np.ndarray
            Input data series
        length : int
            Rolling window length
            
        Returns:
        --------
        np.ndarray : Rolling standard deviation
        """
        std_values = np.zeros_like(data)
        
        for i in range(len(data)):
            start_idx = max(0, i - length + 1)
            window = data[start_idx:i+1]
            std_values[i] = np.std(window, ddof=1) if len(window) > 1 else 0.0
            
        return std_values
    
    @staticmethod
    def gaussian_sample(mean: np.ndarray, cov: np.ndarray, n_samples: int) -> np.ndarray:
        """
        Generate Gaussian random samples.
        
        Parameters:
        -----------
        mean : np.ndarray
            Mean vector (shape: [dim])
        cov : np.ndarray
            Covariance matrix (shape: [dim, dim])
        n_samples : int
            Number of samples to generate
            
        Returns:
        --------
        np.ndarray : Random samples (shape: [n_samples, dim])
        """
        return np.random.multivariate_normal(mean, cov, size=n_samples)
    
    @staticmethod
    def safe_inverse(matrix: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
        """
        Safe matrix inversion with regularization.
        
        Parameters:
        -----------
        matrix : np.ndarray
            Input matrix
        epsilon : float
            Regularization parameter
            
        Returns:
        --------
        np.ndarray : Inverted matrix
        """
        # Add small diagonal term for numerical stability
        regularized = matrix + epsilon * np.eye(matrix.shape[0])
        
        try:
            return linalg.inv(regularized)
        except linalg.LinAlgError:
            # Fallback to pseudo-inverse
            return linalg.pinv(regularized)
    
    @staticmethod
    def l2_norm(vector: np.ndarray) -> float:
        """
        Calculate L2 (Euclidean) norm.
        
        Parameters:
        -----------
        vector : np.ndarray
            Input vector
            
        Returns:
        --------
        float : L2 norm
        """
        return np.sqrt(np.sum(vector ** 2))
    
    @staticmethod
    def empirical_covariance(samples: np.ndarray) -> np.ndarray:
        """
        Calculate empirical covariance from samples.
        
        Parameters:
        -----------
        samples : np.ndarray
            Samples (shape: [n_samples, dim])
            
        Returns:
        --------
        np.ndarray : Covariance matrix (shape: [dim, dim])
        """
        return np.cov(samples, rowvar=False)
