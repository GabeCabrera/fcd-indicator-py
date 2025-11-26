"""
FCD-PSE State Variables
=======================
Core FCD transformation: A_t + B_t → {C_t, X_t} → A'_t
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from .primitives import MathPrimitives
from .kalman import KalmanFilter
from .monte_carlo import MonteCarloEngine


class FCDState:
    """
    FCD State Manager
    
    Manages the complete FCD transformation cycle:
    - A_t: Actual State (observed)
    - B_t: Potential State (distribution)
    - C_t: Internal Resolution (tension)
    - X_t: External Resolution (Bayesian update)
    - A'_t: New Actual State (posterior)
    """
    
    def __init__(self,
                 fast_length: int = 12,
                 slow_length: int = 26,
                 signal_length: int = 9,
                 atr_length: int = 14,
                 vol_length: int = 20,
                 n_paths: int = 1000,
                 obs_noise_scale: float = 0.1,
                 diffusion_scale: float = 1.0,
                 horizon: int = 1,
                 memory_depth: int = 5,
                 memory_lambda: float = 0.25,
                 memory_power: float = 1.0,
                 tension_weights: Optional[List[float]] = None,
                 tension_vol_alpha: float = 0.1,
                 mass_coeffs: Optional[Tuple[float, float, float, float]] = None,
                 asym_coeffs: Optional[Tuple[float, float, float]] = None,
                 tension_ema_alpha: float = 0.2,
                 trend_threshold: float = 0.0005,
                 vol_threshold: float = 1.0,
                 momentum_weight: float = 0.2):
        """
        Initialize FCD State system.
        
        Parameters:
        -----------
        fast_length : int
            Fast EMA period.
        slow_length : int
            Slow EMA period.
        signal_length : int
            Signal line period.
        atr_length : int
            ATR period.
        vol_length : int
            Volatility window for volatility features.
        n_paths : int
            Number of Monte Carlo paths.
        obs_noise_scale : float
            Observation noise for Kalman filter.
        diffusion_scale : float
            Diffusion scaling for Monte Carlo propagation.
        horizon : int
            Forecast horizon (number of steps ahead for B_t / X_t).
        memory_depth : int
            Number of past steps used in the nonlinear memory operator.
        memory_lambda : float
            Exponential decay rate for the memory kernel.
        memory_power : float
            Power applied to lag index in the memory kernel.
        tension_weights : list[float], optional
            Component-wise weights for computing C_t = W ⊙ (B_t − A_t).
        tension_vol_alpha : float
            Volatility scaling coefficient for C_mag.
        mass_coeffs : tuple[float, float, float, float], optional
            Coefficients for causal mass function h_mass.
        asym_coeffs : tuple[float, float, float], optional
            Coefficients for asymmetry function h_asym.
        tension_ema_alpha : float
            Smoothing parameter for tension_ema.
        trend_threshold : float
            Threshold for trend_score in regime classification.
        vol_threshold : float
            Threshold for vol_score in regime classification.
        """
        self.fast_length = fast_length
        self.slow_length = slow_length
        self.signal_length = signal_length
        self.atr_length = atr_length
        self.vol_length = vol_length
        
        # Initialize components
        self.kalman = KalmanFilter(state_dim=3, obs_noise_scale=obs_noise_scale)
        self.mc_engine = MonteCarloEngine(state_dim=3, n_paths=n_paths)
        self.diffusion_scale = diffusion_scale
        self.horizon = max(1, int(horizon))
        
        # Nonlinear memory configuration
        self.memory_depth = max(0, int(memory_depth))
        self.memory_lambda = float(memory_lambda)
        self.memory_power = float(memory_power)
        self.tension_weights = np.array(tension_weights) if tension_weights is not None else np.ones(3)
        if self.tension_weights.shape[0] != 3:
            # Resize/clip to state dimension
            self.tension_weights = np.resize(self.tension_weights, 3)
        self.tension_vol_alpha = max(0.0, float(tension_vol_alpha))
        self.mass_coeffs = mass_coeffs or (0.1, 0.5, 0.3, 0.2)
        self.asym_coeffs = asym_coeffs or (0.0, 1.0, 0.5)
        self.tension_ema_alpha = float(tension_ema_alpha)
        self.trend_threshold = max(0.0, float(trend_threshold))
        self.vol_threshold = max(0.0, float(vol_threshold))
        self.momentum_weight = float(momentum_weight)
        self.skew_window = 30
        self.realized_vol_window = max(10, vol_length)
        self.tension_ema = 0.0
        
        # Regime weights help modulate memory and mass
        self.regime_weight_map = {
            'uptrend_low_vol': 1.0,
            'uptrend_high_vol': 1.2,
            'downtrend_low_vol': 0.9,
            'downtrend_high_vol': 1.1,
            'sideways_low_vol': 0.6
        }
        
        # State history
        self.state_history = []
        self.cov_history = []
        self.actual_history: List[np.ndarray] = []
        self.potential_history: List[np.ndarray] = []
        self.tension_history: List[Tuple[np.ndarray, float]] = []
        self.regime_history: List[str] = []
        self.memory_history: List[np.ndarray] = []
        self.regime_persistence: int = 0
        self.last_regime: Optional[str] = None

    def _memory_kernel(self, k: int) -> float:
        """Exponential decay kernel K_mem(k) = exp(-λ k^p)."""
        return float(np.exp(-self.memory_lambda * (k ** self.memory_power)))

    def _regime_weight(self, regime: str) -> float:
        """Numeric weight for regime-specific modulation."""
        return float(self.regime_weight_map.get(regime, 1.0))

    def _trend_score(self, price: np.ndarray, ema_fast_val: float, ema_slow_val: float, idx: int) -> float:
        """Trend score = signed EMA spread normalized by price."""
        price_t = price[idx] if idx < len(price) else price[-1]
        diff = ema_fast_val - ema_slow_val
        denom = max(abs(price_t), 1e-6)
        return float(np.sign(diff) * (abs(diff) / denom))

    def _realized_vol(self, price: np.ndarray, idx: int) -> float:
        """Realized volatility from returns over a rolling window."""
        start_idx = max(0, idx - self.realized_vol_window + 1)
        window = price[start_idx:idx+1]
        if len(window) < 2:
            return 0.0
        returns = np.diff(window) / window[:-1]
        return float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0

    def _vol_score(self, realized_vol: float, vol_history: List[float]) -> float:
        """Vol score = realized vol divided by median of history."""
        if not vol_history:
            return realized_vol
        median_rv = np.median(vol_history)
        median_rv = median_rv if median_rv > 0 else 1e-6
        return float(realized_vol / median_rv)

    def _skew_of_returns(self, price: np.ndarray, idx: int) -> float:
        """Compute skewness of returns over recent window."""
        start_idx = max(0, idx - self.skew_window + 1)
        window = price[start_idx:idx+1]
        if len(window) < 3:
            return 0.0
        returns = np.diff(window) / window[:-1]
        if len(returns) < 3:
            return 0.0
        mean = np.mean(returns)
        std = np.std(returns) + 1e-6
        skew = np.mean(((returns - mean) / std) ** 3)
        return float(skew)

    def classify_regime(self, trend_score: float, vol_score: float) -> str:
        """
        Symmetric regime classification:
          trend >= +threshold -> uptrend (split by vol)
          trend <= -threshold -> downtrend (split by vol)
          otherwise -> sideways_low_vol (default catch-all)
        """
        high_vol = vol_score >= self.vol_threshold
        if trend_score >= self.trend_threshold:
            return "uptrend_high_vol" if high_vol else "uptrend_low_vol"
        if trend_score <= -self.trend_threshold:
            return "downtrend_high_vol" if high_vol else "downtrend_low_vol"
        return "sideways_low_vol"

    def regime_push_vector(self, regime: str) -> np.ndarray:
        """
        Build a small regime-based adjustment for the transition kernel.
        Momentum component is nudged in the direction implied by trend.
        """
        push = np.zeros(3)
        if "uptrend" in regime:
            push[1] = 0.1
        elif "downtrend" in regime:
            push[1] = -0.1
        return push

    def compute_causal_mass(self, smoothed_tension: float, vol_score: float, regime: str) -> float:
        """h_mass = β0 + β1 * T̄ + β2 * RV + β3 * RegimeIndicator."""
        beta0, beta1, beta2, beta3 = self.mass_coeffs
        regime_weight = self._regime_weight(regime)
        return float(beta0 + beta1 * smoothed_tension + beta2 * vol_score + beta3 * regime_weight)

    def compute_asymmetry(self, trend_score: float, skew: float) -> float:
        """
        κ_t flips sign exactly with trend direction:
          raw = a0 + a1*trend_score + a2*skew
          mag = |raw|
          trend>0 -> +mag, trend<0 -> -mag, flat -> 0
        """
        a0, a1, a2 = self.asym_coeffs if self.asym_coeffs is not None else (0.0, 1.0, 0.5)
        raw = a0 + a1 * trend_score + a2 * skew
        mag = abs(raw)
        if trend_score > 0:
            return float(mag)
        if trend_score < 0:
            return float(-mag)
        return 0.0
    
    def compute_actual_state(self,
                           price: np.ndarray,
                           high: np.ndarray,
                           low: np.ndarray,
                           t: int) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Compute Actual State A_t = [Trend_t, Mom_t, Vol_t].
        
        Parameters:
        -----------
        price : np.ndarray
            Close price series
        high : np.ndarray
            High price series
        low : np.ndarray
            Low price series
        t : int
            Current time index
            
        Returns:
        --------
        Tuple[np.ndarray, Dict] : (A_t state vector, auxiliary metrics)
        """
        # Compute indicator primitives
        ema_fast = MathPrimitives.ema(price[:t+1], self.fast_length)
        ema_slow = MathPrimitives.ema(price[:t+1], self.slow_length)
        macd_line, _, _ = MathPrimitives.macd(
            price[:t+1], 
            self.fast_length, 
            self.slow_length, 
            self.signal_length
        )
        atr = MathPrimitives.atr(high[:t+1], low[:t+1], price[:t+1], self.atr_length)
        
        # State vector at time t
        trend_t = ema_slow[-1]
        mom_t = macd_line[-1]
        vol_t = atr[-1]
        
        A_t = np.array([trend_t, mom_t, vol_t])
        realized_vol = self._realized_vol(price, t)
        vol_score = self._vol_score(realized_vol, [p[2] for p in self.state_history] if self.state_history else [])
        skew = self._skew_of_returns(price, t)
        trend_score = self._trend_score(price, ema_fast[-1], ema_slow[-1], t)
        
        aux = {
            'trend_score': trend_score,
            'vol_score': vol_score,
            'realized_vol': realized_vol,
            'skew': skew
        }
        
        return A_t, aux

    def compute_memory_operator(self, current_state: np.ndarray) -> np.ndarray:
        """
        Nonlinear memory operator:
            M_t = Σ K_mem(k) * φ(A_{t−k}, B_{t−k}, T_{t−k}, regime_{t−k})
        Implemented with decay kernel, regime weighting, and tanh nonlinearity.
        """
        if self.memory_depth == 0 or len(self.actual_history) == 0:
            return np.zeros_like(current_state)
        
        depth = min(self.memory_depth, len(self.actual_history))
        raw = np.zeros_like(current_state, dtype=float)
        weight_sum = 0.0
        
        for k in range(1, depth + 1):
            past_A = self.actual_history[-k]
            past_B = self.potential_history[-k] if len(self.potential_history) >= k else past_A
            past_tension = self.tension_history[-k][1] if len(self.tension_history) >= k else 0.0
            past_regime = self.regime_history[-k] if len(self.regime_history) >= k else "sideways_low_vol"
            
            kernel_w = self._memory_kernel(k)
            regime_w = self._regime_weight(past_regime)
            feature = past_B - past_A
            
            raw += kernel_w * regime_w * (1.0 + past_tension) * feature
            weight_sum += kernel_w * abs(regime_w)

        # Symmetric momentum component on trend dimension to allow downward accumulation
        if len(self.actual_history) > 0:
            price_delta = current_state[0] - self.actual_history[-1][0]
            raw[0] += self.momentum_weight * np.tanh(price_delta)
        
        if weight_sum > 0:
            raw = raw / weight_sum
        
        return np.tanh(raw)
    
    def compute_potential_state(self,
                               A_t: np.ndarray,
                               state_cov: np.ndarray,
                               memory_vec: np.ndarray,
                               causal_mass: float,
                               asymmetry: float,
                               regime_push: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute Potential State B_t using the quasi-Markov transition kernel.
        
        Parameters:
        -----------
        A_t : np.ndarray
            Current actual state
        state_cov : np.ndarray
            State covariance matrix
        memory_vec : np.ndarray
            Nonlinear memory operator M_t
        causal_mass : float
            Causal mass M_t^c
        asymmetry : float
            Asymmetry coefficient κ_t
        regime_push : np.ndarray
            Regime-based adjustment vector
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray, np.ndarray] : (paths, B_mean, B_cov)
        """
        transition_fn = self.mc_engine.quasi_markov_transition(
            memory_vector=memory_vec,
            causal_mass=causal_mass,
            asymmetry=asymmetry,
            regime_push=regime_push,
            tension_scale=1.0 + self.tension_ema,
            horizon=self.horizon
        )
        
        paths = self.mc_engine.propagate_paths(
            current_state=A_t,
            state_cov=state_cov,
            transition_fn=transition_fn,
            horizon=self.horizon,
            diffusion_scale=self.diffusion_scale
        )
        
        B_mean, B_cov = self.mc_engine.estimate_distribution(paths)
        
        return paths, B_mean, B_cov
    
    def compute_internal_resolution(self,
                                   A_t: np.ndarray,
                                   B_mean: np.ndarray,
                                   volatility: float) -> Tuple[np.ndarray, float]:
        """
        Compute Internal Resolution C_t (tension between actual and potential).
        
        Parameters:
        -----------
        A_t : np.ndarray
            Actual state
        B_mean : np.ndarray
            Mean of potential state distribution
            
        Returns:
        --------
        Tuple[np.ndarray, float] : (C_t, C_mag_t)
            - C_t: Tension vector.
            - C_mag_t: Tension magnitude (L2 norm) scaled as:
              ||W ⊙ (B_t − A_t)|| * (1 + α * vol_t).
        """
        C_t = self.tension_weights * (B_mean - A_t)
        C_mag_t = MathPrimitives.l2_norm(C_t) * (1.0 + self.tension_vol_alpha * max(volatility, 0.0))
        
        return C_t, C_mag_t
    
    def compute_external_resolution(self,
                                   B_mean: np.ndarray,
                                   B_cov: np.ndarray,
                                   Obs_next: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute External Resolution X_t via Kalman update.
        
        Parameters:
        -----------
        B_mean : np.ndarray
            Predicted mean from potential state
        B_cov : np.ndarray
            Predicted covariance from potential state
        Obs_next : np.ndarray
            Observed next state
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray] : (X_t, K_t)
            - X_t: External resolution (posterior state)
            - K_t: Kalman gain matrix
        """
        # Compute Kalman gain
        K_t = self.kalman.compute_gain(B_cov)
        
        # Bayesian update
        innovation = Obs_next - B_mean
        X_t = B_mean + K_t @ innovation
        
        return X_t, K_t
    
    def compute_new_actual(self, X_t: np.ndarray) -> np.ndarray:
        """
        Compute New Actual State A'_t.
        
        Parameters:
        -----------
        X_t : np.ndarray
            External resolution
            
        Returns:
        --------
        np.ndarray : A'_t (new actual state)
        """
        return X_t.copy()
    
    def full_fcd_cycle(self,
                      price: np.ndarray,
                      high: np.ndarray,
                      low: np.ndarray,
                      t: int) -> Dict[str, np.ndarray]:
        """
        Execute complete FCD transformation cycle at time t.
        
        Parameters:
        -----------
        price : np.ndarray
            Close price series (up to current time)
        high : np.ndarray
            High price series
        low : np.ndarray
            Low price series
        t : int
            Current time index
            
        Returns:
        --------
        Dict containing:
            - A_t: Actual state.
            - B_paths: Monte Carlo paths.
            - B_mean: Potential state mean.
            - B_cov: Potential state covariance.
            - C_t: Internal resolution vector.
            - C_mag: Internal resolution magnitude.
            - X_t: External resolution.
            - A_prime: New actual state (posterior).
            - K_t: Kalman gain.
            - memory: Nonlinear memory vector M_t.
            - regime: Regime label (4-bucket classifier).
            - causal_mass: Causal mass h_mass(tension_ema, vol, regime).
            - asymmetry: Asymmetry coefficient h_asym(trend, skew).
        """
        # Step 1: Compute A_t and auxiliary metrics
        A_t, aux = self.compute_actual_state(price, high, low, t)
        trend_score = aux['trend_score']
        vol_score = aux['vol_score']
        skew = aux['skew']
        realized_vol = aux['realized_vol']
        
        # Initialize covariance if needed
        state_cov = self.cov_history[-1] if self.cov_history else np.eye(3) * 0.1
        
        # Step 2: Nonlinear memory operator
        memory_vec = self.compute_memory_operator(A_t)
        
        # Step 3: Regime classification
        regime_label = self.classify_regime(trend_score, vol_score)
        regime_push = self.regime_push_vector(regime_label)
        if regime_label == self.last_regime:
            self.regime_persistence += 1
        else:
            self.regime_persistence = 1
            self.last_regime = regime_label
        
        # Step 4: Transition parameters (causal mass, asymmetry)
        causal_mass = self.compute_causal_mass(self.tension_ema, vol_score, regime_label)
        asymmetry = self.compute_asymmetry(trend_score, skew)
        
        # Step 5: Compute B_t (potential state distribution)
        B_paths, B_mean, B_cov = self.compute_potential_state(
            A_t=A_t,
            state_cov=state_cov,
            memory_vec=memory_vec,
            causal_mass=causal_mass,
            asymmetry=asymmetry,
            regime_push=regime_push
        )
        
        # Step 6: Compute C_t (internal resolution) with volatility scaling
        C_t, C_mag = self.compute_internal_resolution(A_t, B_mean, volatility=A_t[2])
        self.tension_ema = (1 - self.tension_ema_alpha) * self.tension_ema + self.tension_ema_alpha * C_mag
        
        # Update causal mass after new tension value
        causal_mass = self.compute_causal_mass(self.tension_ema, vol_score, regime_label)
        
        # Step 7: Compute observed next state at horizon
        if t + self.horizon < len(price):
            Obs_next, _ = self.compute_actual_state(price, high, low, t + self.horizon)
        else:
            # Use predicted mean if no future observation
            Obs_next = B_mean
        
        # Step 8: Compute X_t (external resolution via Kalman update)
        X_t, K_t = self.compute_external_resolution(B_mean, B_cov, Obs_next)
        
        # Step 9: Compute A'_t (new actual state)
        A_prime = self.compute_new_actual(X_t)
        
        # Update history
        self.state_history.append(A_t)
        self.cov_history.append(B_cov)
        self.actual_history.append(A_t)
        self.potential_history.append(B_mean)
        self.tension_history.append((C_t, C_mag))
        self.regime_history.append(regime_label)
        self.memory_history.append(memory_vec)
        
        tension_regime = 'high' if C_mag >= 1.0 else 'low'
        
        return {
            'A_t': A_t,
            'B_paths': B_paths,
            'B_mean': B_mean,
            'B_cov': B_cov,
            'C_t': C_t,
            'C_mag': C_mag,
            'X_t': X_t,
            'A_prime': A_prime,
            'K_t': K_t,
            'memory': memory_vec,
            'tension_regime': tension_regime,
            'regime_persistence': self.regime_persistence,
            'regime_history_length': len(self.regime_history),
            'trend_score': trend_score,
            'vol_score': vol_score,
            'skew': skew,
            'realized_vol': realized_vol,
            'regime': regime_label,
            'causal_mass': causal_mass,
            'asymmetry': asymmetry
        }
    
    def reset(self):
        """Reset state history and tension statistics."""
        self.state_history = []
        self.cov_history = []
        self.actual_history = []
        self.potential_history = []
        self.tension_history = []
        self.regime_history = []
        self.memory_history = []
        self.tension_ema = 0.0
        self.regime_persistence = 0
        self.last_regime = None
