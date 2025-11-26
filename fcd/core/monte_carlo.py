"""
FCD-PSE Monte Carlo Simulation Engine
======================================
Generates potential state distributions through stochastic path simulation.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Callable
from .primitives import MathPrimitives


class MonteCarloEngine:
    """Monte Carlo path generator for potential state distribution B_t."""
    
    def __init__(self, 
                 state_dim: int = 3,
                 n_paths: int = 1000,
                 random_seed: Optional[int] = None):
        """
        Initialize Monte Carlo engine.
        
        Parameters:
        -----------
        state_dim : int
            Dimension of state vector
        n_paths : int
            Number of simulation paths
        random_seed : int, optional
            Random seed for reproducibility
        """
        self.state_dim = state_dim
        self.n_paths = n_paths
        
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def generate_paths(self,
                      current_state: np.ndarray,
                      state_cov: np.ndarray,
                      transition_fn: Optional[Callable] = None,
                      drift: Optional[np.ndarray] = None,
                      diffusion_scale: float = 1.0) -> np.ndarray:
        """
        Generate Monte Carlo paths for next state.
        
        Parameters:
        -----------
        current_state : np.ndarray
            Current state A_t (shape: [state_dim])
        state_cov : np.ndarray
            State covariance matrix (shape: [state_dim, state_dim])
        transition_fn : Callable, optional
            Custom transition function (state -> next_state)
        drift : np.ndarray, optional
            Deterministic drift term (default: zeros)
        diffusion_scale : float
            Scaling factor for random diffusion
            
        Returns:
        --------
        np.ndarray : Simulated paths (shape: [n_paths, state_dim])
        """
        if drift is None:
            drift = np.zeros(self.state_dim)
        
        # Generate random shocks from covariance
        shocks = MathPrimitives.gaussian_sample(
            mean=np.zeros(self.state_dim),
            cov=state_cov * diffusion_scale,
            n_samples=self.n_paths
        )
        
        if transition_fn is None:
            # Default: random walk with drift
            paths = current_state + drift + shocks
        else:
            # Custom transition model
            paths = np.array([
                transition_fn(current_state) + shock 
                for shock in shocks
            ])
        
        return paths

    def propagate_paths(self,
                        current_state: np.ndarray,
                        state_cov: np.ndarray,
                        transition_fn: Callable[[np.ndarray], np.ndarray],
                        horizon: int = 1,
                        diffusion_scale: float = 1.0) -> np.ndarray:
        """
        Propagate Monte Carlo paths across a configurable horizon.
        
        This implements the quasi-Markov roll-forward described in the updated
        FCD logic: we embed memory and regime into the transition_fn and step
        the system forward H times.
        """
        # Start all paths at the current state
        paths = np.repeat(current_state[None, :], self.n_paths, axis=0)
        
        for _ in range(max(1, horizon)):
            shocks = MathPrimitives.gaussian_sample(
                mean=np.zeros(self.state_dim),
                cov=state_cov * diffusion_scale,
                n_samples=self.n_paths
            )
            proposals = np.array([transition_fn(path) for path in paths])
            paths = proposals + shocks
        
        return paths
    
    def estimate_distribution(self, 
                            paths: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate mean and covariance from Monte Carlo paths.
        
        Parameters:
        -----------
        paths : np.ndarray
            Simulated paths (shape: [n_paths, state_dim])
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray] : (mean, covariance)
        """
        mean = np.mean(paths, axis=0)
        cov = MathPrimitives.empirical_covariance(paths)
        
        return mean, cov
    
    def transition_model_ar1(self,
                            current_state: np.ndarray,
                            ar_coeff: float = 0.95,
                            mean_reversion_target: Optional[np.ndarray] = None) -> np.ndarray:
        """
        AR(1) transition model with mean reversion.
        
        Parameters:
        -----------
        current_state : np.ndarray
            Current state
        ar_coeff : float
            Autoregressive coefficient (0 < ar_coeff < 1)
        mean_reversion_target : np.ndarray, optional
            Long-term mean (default: zeros)
            
        Returns:
        --------
        np.ndarray : Next state mean under AR(1) model
        """
        if mean_reversion_target is None:
            mean_reversion_target = np.zeros(self.state_dim)
        
        next_state = ar_coeff * current_state + (1 - ar_coeff) * mean_reversion_target
        
        return next_state
    
    def transition_model_momentum(self,
                                 current_state: np.ndarray,
                                 previous_state: np.ndarray,
                                 momentum_coeff: float = 0.5,
                                 decay: float = 0.95) -> np.ndarray:
        """
        Momentum-based transition model.
        
        Parameters:
        -----------
        current_state : np.ndarray
            Current state A_t
        previous_state : np.ndarray
            Previous state A_{t-1}
        momentum_coeff : float
            Momentum strength
        decay : float
            Momentum decay factor
            
        Returns:
        --------
        np.ndarray : Next state mean with momentum
        """
        momentum = current_state - previous_state
        next_state = current_state + momentum_coeff * momentum * decay
        
        return next_state
    
    def transition_model_volatility_adjusted(self,
                                            current_state: np.ndarray,
                                            volatility_idx: int = 2,
                                            vol_threshold: float = 1.0) -> np.ndarray:
        """
        Volatility-adjusted transition model.
        
        High volatility leads to mean reversion.
        Low volatility allows trend continuation.
        
        Parameters:
        -----------
        current_state : np.ndarray
            Current state [Trend, Mom, Vol]
        volatility_idx : int
            Index of volatility in state vector
        vol_threshold : float
            Volatility threshold for regime switching
            
        Returns:
        --------
        np.ndarray : Next state mean
        """
        vol = current_state[volatility_idx]
        
        if vol > vol_threshold:
            # High volatility: mean reversion
            reversion_coeff = 0.3
            next_state = current_state * reversion_coeff
        else:
            # Low volatility: trend continuation
            continuation_coeff = 1.05
            next_state = current_state * continuation_coeff
        
        return next_state

    def quasi_markov_transition(self,
                                memory_vector: np.ndarray,
                                causal_mass: float,
                                asymmetry: float,
                                regime_push: np.ndarray,
                                tension_scale: float = 1.0,
                                horizon: int = 1) -> Callable[[np.ndarray], np.ndarray]:
        """
        Build a transition function that encodes memory, causal mass, asymmetry,
        and regime bias as described by the quasi-Markov kernel.
        
        Parameters:
        -----------
        memory_vector : np.ndarray
            Nonlinear memory M_t (shape: [state_dim])
        causal_mass : float
            Causal mass M_t^c (higher = more inertia)
        asymmetry : float
            Asymmetry coefficient κ_t (positive biases upward momentum component)
        regime_push : np.ndarray
            Regime-based adjustment vector (shape: [state_dim])
        tension_scale : float
            Scalar factor applied to the memory vector (captures tension influence)
        horizon : int
            Forecast horizon; used to scale transition strength
        
        Returns:
        --------
        Callable[[np.ndarray], np.ndarray] : transition function for path propagation
        """
        inertia = 1.0 / (1.0 + max(causal_mass, 0.0))
        horizon_scale = max(1, horizon)
        asym_push = np.zeros_like(memory_vector)
        if len(memory_vector) > 1:
            asym_push[1] = asymmetry  # bias the momentum dimension
        
        def _transition(state: np.ndarray) -> np.ndarray:
            # Memory acts as attractor; causal mass damps movement; asymmetry biases momentum
            memory_pull = inertia * tension_scale * memory_vector
            adjustment = (memory_pull + regime_push + asym_push) / float(horizon_scale)
            return state + adjustment
        
        return _transition
    
    def compute_path_weights(self,
                           paths: np.ndarray,
                           target_state: np.ndarray,
                           temperature: float = 1.0,
                           asymmetry: Optional[float] = None,
                           directional_index: int = 1,
                           regime_bias: Optional[float] = None) -> np.ndarray:
        """
        Compute path weights using a Boltzmann-style distribution.
        
        Parameters:
        -----------
        paths : np.ndarray
            Simulated paths (shape: [n_paths, state_dim])
        target_state : np.ndarray
            Target state for weighting (e.g., A'_t or B_mean)
        temperature : float
            Temperature parameter (higher = more uniform weights)
        asymmetry : float, optional
            Directional bias applied along `directional_index`;
            positive values tilt probability mass toward paths with
            larger component values in that dimension.
        directional_index : int
            Index of the state component used for asymmetry bias.
        regime_bias : float, optional
            Regime-specific scaling factor applied to the distance
            scale (e.g., to flatten or sharpen distributions).
            
        Returns:
        --------
        np.ndarray : Normalized path probabilities (shape: [n_paths])
        """
        temperature = max(1e-6, float(temperature))
        paths = np.asarray(paths)
        target_state = np.asarray(target_state)
        if paths.ndim != 2:
            raise ValueError("paths must have shape [n_paths, state_dim]")
        if target_state.shape[0] != paths.shape[1]:
            raise ValueError("target_state dimension must match paths.shape[1]")
        
        # Ensure directional index is in range
        directional_index = int(directional_index)
        directional_index = max(0, min(directional_index, paths.shape[1] - 1))
        
        # Calculate distances to target (energy)
        distances = np.array([
            MathPrimitives.l2_norm(path - target_state)
            for path in paths
        ])
        
        # Optional directional / regime biasing (asymmetry term)
        if asymmetry is not None:
            direction_component = paths[:, directional_index]
            distances = distances - asymmetry * direction_component
        
        if regime_bias is not None:
            distances = distances * (1.0 + regime_bias)
        
        # Boltzmann-style weights with temperature
        logits = -distances / temperature
        logits = logits - np.max(logits)  # numerical stability
        weights = np.exp(logits)
        
        # Normalize to probabilities
        weight_sum = np.sum(weights)
        if weight_sum == 0 or not np.isfinite(weight_sum):
            # Fallback to uniform distribution if weights are invalid
            probabilities = np.ones(len(weights)) / len(weights)
        else:
            probabilities = weights / weight_sum
        
        return probabilities
    
    def stratified_sampling(self,
                          paths: np.ndarray,
                          n_bins: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Stratified sampling to ensure diversity in path space.
        
        Parameters:
        -----------
        paths : np.ndarray
            Simulated paths
        n_bins : int
            Number of stratification bins per dimension
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray] : (stratified_paths, bin_indices)
        """
        # Simple stratification: divide first dimension into bins
        first_dim = paths[:, 0]
        bin_edges = np.linspace(first_dim.min(), first_dim.max(), n_bins + 1)
        bin_indices = np.digitize(first_dim, bin_edges[:-1])
        
        # Sample one path per bin
        stratified_paths = []
        for bin_idx in range(1, n_bins + 1):
            bin_mask = (bin_indices == bin_idx)
            if np.any(bin_mask):
                bin_paths = paths[bin_mask]
                selected_path = bin_paths[np.random.randint(len(bin_paths))]
                stratified_paths.append(selected_path)
        
        return np.array(stratified_paths), bin_indices
