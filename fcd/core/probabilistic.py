"""
FCD-PSE Probabilistic Prediction Module
========================================
Generates probability distributions for next-state predictions.
"""

import numpy as np
from typing import Tuple, Dict, List, Optional
from .primitives import MathPrimitives
from .monte_carlo import MonteCarloEngine


class ProbabilisticPredictor:
    """
    Probabilistic next-state prediction system.
    
    Generates p(A'_t | A_t, B_t, C_t, X_t) using:
    - Kalman posterior distribution
    - Reweighted Monte Carlo paths
    - Temperature-based softmax weighting
    """
    
    def __init__(self, temperature: float = 1.0):
        """
        Initialize probabilistic predictor.
        
        Parameters:
        -----------
        temperature : float
            Temperature for Boltzmann weighting (higher = more uniform)
        """
        self.temperature = temperature
        self.mc_engine = MonteCarloEngine(state_dim=3)
    
    def kalman_posterior_distribution(self,
                                     A_prime: np.ndarray,
                                     posterior_cov: np.ndarray,
                                     n_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate samples from Kalman posterior distribution.
        
        Parameters:
        -----------
        A_prime : np.ndarray
            Posterior mean (A'_t)
        posterior_cov : np.ndarray
            Posterior covariance (I - K)*PredCov
        n_samples : int
            Number of samples to generate
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray] : (samples, uniform_weights)
        """
        samples = MathPrimitives.gaussian_sample(
            mean=A_prime,
            cov=posterior_cov,
            n_samples=n_samples
        )
        
        # Uniform weights for Gaussian samples
        weights = np.ones(n_samples) / n_samples
        
        return samples, weights
    
    def reweight_mc_paths(self,
                         paths: np.ndarray,
                         A_prime: np.ndarray,
                         temperature: Optional[float] = None,
                         asymmetry: Optional[float] = None,
                         directional_index: int = 1,
                         regime_bias: Optional[float] = None) -> np.ndarray:
        """
        Reweight Monte Carlo paths based on distance to A'_t.
        
        Parameters:
        -----------
        paths : np.ndarray
            Monte Carlo paths (shape: [n_paths, state_dim])
        A_prime : np.ndarray
            Target state for reweighting (typically A'_t)
        temperature : float, optional
            Temperature parameter (uses self.temperature if None).
            Higher values yield more uniform weights.
        asymmetry : float, optional
            Directional asymmetry coefficient passed through to
            `MonteCarloEngine.compute_path_weights`.
        directional_index : int
            State component index used for asymmetry bias.
        regime_bias : float, optional
            Regime-specific bias factor passed through to
            `MonteCarloEngine.compute_path_weights`.
            
        Returns:
        --------
        np.ndarray : Path probabilities (shape: [n_paths])
        """
        if temperature is None:
            temperature = self.temperature
        
        probabilities = self.mc_engine.compute_path_weights(
            paths=paths,
            target_state=A_prime,
            temperature=temperature,
            asymmetry=asymmetry,
            directional_index=directional_index,
            regime_bias=regime_bias
        )
        
        return probabilities
    
    def predict_distribution(self,
                           paths: np.ndarray,
                           A_prime: np.ndarray,
                           method: str = 'reweighted',
                           posterior_cov: Optional[np.ndarray] = None,
                           temperature: Optional[float] = None,
                           asymmetry: Optional[float] = None,
                           regime_bias: Optional[float] = None,
                           directional_index: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predictive distribution for next state.
        
        Parameters:
        -----------
        paths : np.ndarray
            Monte Carlo paths from B_t.
        A_prime : np.ndarray
            Posterior state estimate (A'_t).
        method : str
            'reweighted' (reweight MC paths) or 'kalman' (Gaussian posterior).
        posterior_cov : np.ndarray, optional
            Posterior covariance (required for 'kalman' method).
        temperature : float, optional
            Temperature for reweighting; falls back to self.temperature.
        asymmetry : float, optional
            Directional asymmetry coefficient for MC path weighting.
        regime_bias : float, optional
            Regime-specific bias factor for MC path weighting.
        directional_index : int
            State component index used for asymmetry bias.
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray] : (samples, probabilities)
        """
        if method == 'reweighted':
            probabilities = self.reweight_mc_paths(
                paths, A_prime,
                temperature=temperature,
                asymmetry=asymmetry,
                directional_index=directional_index,
                regime_bias=regime_bias
            )
            return paths, probabilities
        
        elif method == 'kalman':
            if posterior_cov is None:
                raise ValueError("posterior_cov required for kalman method")
            return self.kalman_posterior_distribution(A_prime, posterior_cov, n_samples=len(paths))
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def distribution_statistics(self,
                              samples: np.ndarray,
                              probabilities: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute statistics of the predictive distribution.
        
        Parameters:
        -----------
        samples : np.ndarray
            Sample paths (shape: [n_samples, state_dim])
        probabilities : np.ndarray
            Sample probabilities (shape: [n_samples])
            
        Returns:
        --------
        Dict containing:
            - mean: Weighted mean
            - std: Weighted standard deviation
            - median: Weighted median approximation
            - quantiles: [5%, 25%, 50%, 75%, 95%]
        """
        # Weighted mean
        mean = np.average(samples, weights=probabilities, axis=0)
        
        # Weighted variance and std
        variance = np.average((samples - mean) ** 2, weights=probabilities, axis=0)
        std = np.sqrt(variance)
        
        # Approximate weighted quantiles
        quantiles = {}
        for q in [0.05, 0.25, 0.50, 0.75, 0.95]:
            # Simple approximation: use cumulative weights
            sorted_indices = np.argsort(samples[:, 0])  # Sort by first dimension
            cumulative_weights = np.cumsum(probabilities[sorted_indices])
            quantile_idx = np.searchsorted(cumulative_weights, q)
            quantile_idx = min(quantile_idx, len(samples) - 1)
            quantiles[f'q{int(q*100)}'] = samples[sorted_indices[quantile_idx]]
        
        return {
            'mean': mean,
            'std': std,
            'median': quantiles['q50'],
            'quantiles': quantiles
        }
    
    def directional_bias(self,
                        samples: np.ndarray,
                        probabilities: np.ndarray,
                        component_idx: int = 1) -> Dict[str, float]:
        """
        Compute directional bias for a specific component (e.g., momentum).
        
        Computes probability of component increasing vs decreasing relative
        to the weighted mean (not relative to zero).
        
        Parameters:
        -----------
        samples : np.ndarray
            Sample paths
        probabilities : np.ndarray
            Sample probabilities
        component_idx : int
            Index of component to analyze (default: 1 for momentum)
            
        Returns:
        --------
        Dict containing:
            - upward_prob: Probability of above-mean component values
            - downward_prob: Probability of below-mean component values
            - expected_change: Weighted mean of component
        """
        component_values = samples[:, component_idx]
        
        # Compute weighted mean
        weighted_mean = np.average(component_values, weights=probabilities)
        
        # Check if values are above or below the weighted mean
        # This gives us directional probability relative to expected value
        upward_mask = component_values > weighted_mean
        downward_mask = component_values < weighted_mean
        
        upward_prob = np.sum(probabilities[upward_mask])
        downward_prob = np.sum(probabilities[downward_mask])
        
        return {
            'upward_prob': upward_prob,
            'downward_prob': downward_prob,
            'expected_change': weighted_mean,
            'bias': upward_prob - downward_prob
        }
    
    def distribution_entropy(self, probabilities: np.ndarray) -> float:
        """
        Compute entropy of probability distribution.
        
        Higher entropy = more uncertainty.
        
        Parameters:
        -----------
        probabilities : np.ndarray
            Probability distribution
            
        Returns:
        --------
        float : Entropy (in nats)
        """
        # Filter out zero probabilities
        p_nonzero = probabilities[probabilities > 0]
        entropy = -np.sum(p_nonzero * np.log(p_nonzero))
        return entropy
    
    def confidence_interval(self,
                          samples: np.ndarray,
                          probabilities: np.ndarray,
                          confidence_level: float = 0.95) -> Dict[str, np.ndarray]:
        """
        Compute confidence intervals for each state component.
        
        Parameters:
        -----------
        samples : np.ndarray
            Sample paths
        probabilities : np.ndarray
            Sample probabilities
        confidence_level : float
            Confidence level (default: 0.95)
            
        Returns:
        --------
        Dict containing:
            - lower: Lower confidence bounds
            - upper: Upper confidence bounds
        """
        alpha = 1 - confidence_level
        lower_quantile = alpha / 2
        upper_quantile = 1 - alpha / 2
        
        state_dim = samples.shape[1]
        lower_bounds = np.zeros(state_dim)
        upper_bounds = np.zeros(state_dim)
        
        for dim in range(state_dim):
            sorted_indices = np.argsort(samples[:, dim])
            cumulative_weights = np.cumsum(probabilities[sorted_indices])
            
            lower_idx = np.searchsorted(cumulative_weights, lower_quantile)
            upper_idx = np.searchsorted(cumulative_weights, upper_quantile)
            
            lower_idx = min(lower_idx, len(samples) - 1)
            upper_idx = min(upper_idx, len(samples) - 1)
            
            lower_bounds[dim] = samples[sorted_indices[lower_idx], dim]
            upper_bounds[dim] = samples[sorted_indices[upper_idx], dim]
        
        return {
            'lower': lower_bounds,
            'upper': upper_bounds
        }
