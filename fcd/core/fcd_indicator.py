"""
FCD-PSE Indicator Main Interface
=================================
Complete FCD-PSE indicator with signal generation and output.
"""

import numpy as np
from typing import Dict, Optional, Tuple
from .fcd_state import FCDState
from .probabilistic import ProbabilisticPredictor
from .multi_scale import MultiScaleFCD
from .btc_mode_config import BTC_CONFIG


class FCDIndicator:
    """
    Complete FCD-PSE Indicator System
    
    Provides:
    - Single-timeframe FCD analysis
    - Multi-timeframe coherence
    - Trading signals
    - Probabilistic forecasts
    """
    
    def __init__(self,
                 fast_length: int = 12,
                 slow_length: int = 26,
                 signal_length: int = 9,
                 atr_length: int = 14,
                 vol_length: int = 20,
                 n_paths: int = 1000,
                 temperature: float = 1.0,
                 enable_multi_scale: bool = False,
                 timeframes: Optional[list] = None,
                 horizon: int = 1,
                 memory_depth: int = 5,
                 memory_lambda: float = 0.25,
                 memory_power: float = 1.0,
                 tension_weights: Optional[list] = None,
                 tension_vol_alpha: float = 0.1,
                 mass_coeffs: Optional[Tuple[float, float, float, float]] = None,
                 asym_coeffs: Optional[Tuple[float, float, float]] = None,
                 long_threshold: float = 0.1,
                 short_threshold: float = 0.1,
                 trend_threshold: float = 0.0005,
                 vol_threshold: float = 1.0,
                 interval: Optional[str] = None,
                 allow_shorts: bool = True):
        """
        Initialize FCD indicator.
        
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
            Volatility window.
        n_paths : int
            Number of Monte Carlo paths for the Monte Carlo engine.
        temperature : float
            Temperature for probabilistic prediction.
        enable_multi_scale : bool
            Enable multi-timeframe analysis.
        timeframes : list, optional
            List of timeframes for multi-scale.
        horizon : int
            Forecast horizon for the FCD state cycle.
        memory_depth : int
            Depth of the nonlinear memory operator.
        memory_lambda : float
            Decay rate for the memory kernel.
        memory_power : float
            Power applied to lags in the memory kernel.
        tension_weights : list, optional
            Component-wise weights for internal tension C_t.
        tension_vol_alpha : float
            Volatility scaling coefficient for C_mag.
        mass_coeffs : tuple, optional
            Coefficients for causal mass function.
        asym_coeffs : tuple, optional
            Coefficients for asymmetry function.
        long_threshold : float
            Threshold for activating long signals.
        short_threshold : float
            Threshold for activating short signals.
        trend_threshold : float
            Trend score threshold for regime classification.
        vol_threshold : float
            Volatility score threshold for regime classification.
        interval : str, optional
            Data interval label used for profile-aware gating.
        allow_shorts : bool
            Global flag controlling whether short signals are permitted.
        """
        # Single-timeframe FCD
        self.fcd = FCDState(
            fast_length=fast_length,
            slow_length=slow_length,
            signal_length=signal_length,
            atr_length=atr_length,
            vol_length=vol_length,
            n_paths=n_paths,
            horizon=horizon,
            memory_depth=memory_depth,
            memory_lambda=memory_lambda,
            memory_power=memory_power,
            tension_weights=tension_weights,
            tension_vol_alpha=tension_vol_alpha,
            mass_coeffs=mass_coeffs,
            asym_coeffs=asym_coeffs,
            trend_threshold=trend_threshold,
            vol_threshold=vol_threshold
        )
        
        # Probabilistic predictor
        self.predictor = ProbabilisticPredictor(temperature=temperature)
        self.temperature = temperature
        self.allow_shorts = bool(allow_shorts)
        self.interval = interval.lower() if interval else None
        self.vol_proxy_history: list = []
        self.vol_window = 20
        self.low_vol_threshold = 0.005
        self.high_vol_threshold = 0.02
        
        # Multi-scale system
        self.enable_multi_scale = enable_multi_scale
        if enable_multi_scale:
            self.multi_scale = MultiScaleFCD(timeframes=timeframes)
        else:
            self.multi_scale = None
        
        # Signal history
        self.signals_history = []
        
        # Normalization parameters
        self.c_mag_history = []  # Rolling buffer for C_mag normalization
        self.normalization_window = 200  # Window for rolling median
        self.volatility_history = []  # For volatility normalization
        
        # Regime-aware signal permissions
        self.regime_permissions = {
            'uptrend_low_vol': {'long': True, 'short': False, 'scale': 1.0},
            'uptrend_high_vol': {'long': True, 'short': False, 'scale': 0.8},
            'downtrend_low_vol': {'long': False, 'short': True, 'scale': 1.0},
            'downtrend_high_vol': {'long': False, 'short': True, 'scale': 1.0},
            'sideways_low_vol': {'long': False, 'short': False, 'scale': 0.0},
            'sideways_high_vol': {'long': False, 'short': False, 'scale': 0.0}
        }
        self.long_threshold = long_threshold
        self.short_threshold = short_threshold
        self.vol_proxy_history: list = []
        self.vol_window = 20
        self.low_vol_threshold = 0.005
        self.high_vol_threshold = 0.02
    
    def _regime_bias(self, regime: str) -> float:
        """Small regime-dependent bias for path weighting."""
        if regime.startswith("sideways"):
            return 0.1
        if regime.startswith("downtrend"):
            return 0.05
        return 0.0
    
    def normalize_c_mag(self, c_mag_raw: float, volatility: float) -> Tuple[float, Dict]:
        """
        Normalize C_mag using rolling median approach.
        
        This produces C_norm in realistic ranges:
        - Typical: 0.1 to 2.0
        - Extreme: > 3.0
        - Coherence = 1/(1+C_norm) will range 0.25 to 0.95
        
        Parameters:
        -----------
        c_mag_raw : float
            Raw C_mag from FCD computation (L2 norm of tension)
        volatility : float
            Current volatility state
            
        Returns:
        --------
        Tuple[float, Dict] : (C_norm, diagnostics)
        """
        # Store raw value
        self.c_mag_history.append(c_mag_raw)
        self.volatility_history.append(volatility)
        
        # Keep window size bounded
        if len(self.c_mag_history) > self.normalization_window:
            self.c_mag_history.pop(0)
        if len(self.volatility_history) > self.normalization_window:
            self.volatility_history.pop(0)
        
        # Compute normalization factor (rolling median)
        if len(self.c_mag_history) >= 20:  # Minimum warmup
            c_mag_median = np.median(self.c_mag_history)
            # Prevent division by zero
            normalization_factor = max(c_mag_median, 1.0)
        else:
            # During warmup, use volatility-based estimate
            normalization_factor = max(volatility * 10, 1.0)
        
        # Normalize: C_norm = C_raw / (1 + median(C_raw))
        # This produces typical values in 0.1-2.0 range
        c_norm = c_mag_raw / (1.0 + normalization_factor)
        
        # Diagnostics
        diagnostics = {
            'c_mag_raw': c_mag_raw,
            'c_norm': c_norm,
            'normalization_factor': normalization_factor,
            'c_mag_median': np.median(self.c_mag_history) if len(self.c_mag_history) >= 20 else None,
            'warmup_complete': len(self.c_mag_history) >= self.normalization_window
        }
        
        return c_norm, diagnostics
    
    def classify_btc_regime(self, coherence: float, tv_product: float, vol_regime: str) -> str:
        """
        Classify BTC market regime based on FCD metrics.
        
        Parameters:
        -----------
        coherence : float
            Normalized coherence value
        tv_product : float
            Normalized TV product value
        vol_regime : str
            Volatility regime ('low', 'medium', 'high')
            
        Returns:
        --------
        str : Regime classification ('drift', 'becoming', 'chop')
        """
        # Drift: High coherence, low TV, appropriate volatility
        if (coherence >= BTC_CONFIG.COH_DRIFT_HIGH and 
            tv_product <= BTC_CONFIG.TV_LOW_MAX and 
            vol_regime in BTC_CONFIG.ALLOWED_VOL_DRIFT):
            return "drift"
        
        # Becoming: Moderate coherence and TV, appropriate volatility
        if (BTC_CONFIG.COH_BECO_MIN <= coherence <= BTC_CONFIG.COH_BECO_MAX and 
            BTC_CONFIG.TV_BECO_MIN <= tv_product <= BTC_CONFIG.TV_BECO_MAX and 
            vol_regime in BTC_CONFIG.ALLOWED_VOL_BECO):
            return "becoming"
        
        # Default: Chop
        return "chop"
    
    def compute(self,
                price: np.ndarray,
                high: np.ndarray,
                low: np.ndarray,
                t: int) -> Dict:
        """
        Compute FCD indicator at time t.
        
        Parameters:
        -----------
        price : np.ndarray
            Close price series.
        high : np.ndarray
            High price series.
        low : np.ndarray
            Low price series.
        t : int
            Current time index.
            
        Returns:
        --------
        Dict : Complete FCD output including core state
               (A_t, B_t, C_t, X_t, A'_t), nonlinear memory,
               causal mass, asymmetry, regimes, probabilities,
               and regime-aware trading signals.
        """
        # Core FCD transformation (encodes memory, mass, asymmetry, regime)
        fcd_result = self.fcd.full_fcd_cycle(price, high, low, t)
        
        tension_raw = fcd_result['C_mag']  # already weighted per new definition
        volatility = float(fcd_result['A_prime'][2])  # Volatility component
        vol_proxy = fcd_result.get('A_t', [0, 0, volatility])[2]
        self.vol_proxy_history.append(vol_proxy / max(price[t], 1e-6))
        if len(self.vol_proxy_history) > self.vol_window:
            self.vol_proxy_history.pop(0)
        vol_level = float(np.mean(self.vol_proxy_history)) if self.vol_proxy_history else 0.0
        if vol_level < self.low_vol_threshold:
            vol_regime_local = "low_vol"
        elif vol_level < self.high_vol_threshold:
            vol_regime_local = "mid_vol"
        else:
            vol_regime_local = "high_vol"
        
        # Normalize C_mag for coherence-style metrics (backward compatibility)
        c_norm, norm_diagnostics = self.normalize_c_mag(tension_raw, volatility)
        coherence = 1.0 / (1.0 + c_norm)
        
        vol_median = np.median(self.volatility_history) if len(self.volatility_history) >= 20 else volatility
        tv_product = (c_norm * volatility) / (1.0 + vol_median)
        vol_z = volatility / max(vol_median, 1e-6) if vol_median is not None else 0.0
        
        A_prime = fcd_result['A_prime']
        B_paths = fcd_result['B_paths']
        
        # Probabilistic prediction with asymmetry-aware weighting
        regime_bias = self._regime_bias(fcd_result.get('regime', 'sideways_low_vol'))
        samples, probabilities = self.predictor.predict_distribution(
            paths=B_paths,
            A_prime=A_prime,
            method='reweighted',
            temperature=self.temperature,
            asymmetry=fcd_result.get('asymmetry'),
            regime_bias=regime_bias,
            directional_index=1
        )
        
        dist_stats = self.predictor.distribution_statistics(samples, probabilities)
        directional = self.predictor.directional_bias(samples, probabilities, component_idx=1)
        confidence = self.predictor.confidence_interval(samples, probabilities)
        entropy = self.predictor.distribution_entropy(probabilities)
        path_var = None
        try:
            if B_paths is not None and len(B_paths) > 0:
                direction_index = 1
                if B_paths.ndim == 3:  # (paths, horizon, state_dim)
                    path_var = float(np.var(B_paths[:, -1, direction_index]))
                else:
                    path_var = float(np.var(B_paths[:, direction_index]))
        except Exception:
            path_var = None
        
        # Volatility normalization/regime
        vol_median = np.median(self.volatility_history) if len(self.volatility_history) >= 20 else volatility
        normalized_volatility = volatility / (1.0 + vol_median) if vol_median > 0 else volatility
        # Map local proxy regime to legacy label for compatibility
        if vol_regime_local == "low_vol":
            vol_regime = 'low'
        elif vol_regime_local == "mid_vol":
            vol_regime = 'medium'
        else:
            vol_regime = 'high'
        
        regime_label = fcd_result.get('regime', 'sideways_low_vol')
        fcd_result['path_var'] = path_var
        fcd_result['vol_z'] = vol_z
        fcd_result['vol_regime'] = vol_regime_local
        signals = self.generate_signals(
            fcd_result=fcd_result,
            directional=directional,
            dist_stats=dist_stats,
            coherence=coherence,
            tension_norm=c_norm
        )
        # Track latest signals for optional downstream analysis
        self.signals_history.append(signals)
        
        legacy_regime = self.classify_btc_regime(coherence, tv_product, vol_regime)
        
        return {
            # Core FCD state
            'A_t': fcd_result['A_t'],
            'B_mean': fcd_result['B_mean'],
            'C_t': fcd_result['C_t'],
            'C_mag_raw': tension_raw,
            'X_t': fcd_result['X_t'],
            'A_prime': A_prime,
            'path_var': path_var,
            
            # Updated core metrics
            'memory': fcd_result.get('memory'),
            'causal_mass': fcd_result.get('causal_mass'),
            'asymmetry': fcd_result.get('asymmetry'),
            'regime': regime_label,
            'regime_permissions': self.regime_permissions.get(regime_label, {'long': True, 'short': False, 'scale': 1.0}),
            'allow_shorts': self.allow_shorts,
            'trend_score': fcd_result.get('trend_score'),
            'vol_score': fcd_result.get('vol_score'),
            'skew': fcd_result.get('skew'),
            'tension_regime': fcd_result.get('tension_regime'),
            'regime_persistence': fcd_result.get('regime_persistence'),
            'vol_regime': vol_regime_local,
            'regime_history_length': fcd_result.get('regime_history_length'),
            
            # NORMALIZED VALUES (Primary)
            'normalized_C_mag': c_norm,
            'normalized_coherence': coherence,
            'normalized_tv_product': tv_product,
            'normalized_volatility': normalized_volatility,
            'vol_z': vol_z,
            
            # Backward compatibility aliases (point to normalized)
            'C_mag': c_norm,
            'C_magnitude': c_norm,
            'coherence': coherence,
            'tv_product': tv_product,
            'volatility': normalized_volatility,
            
            'normalization_diagnostics': norm_diagnostics,
            
            # Regime classification
            'vol_regime': vol_regime,
            'legacy_regime': legacy_regime,
            
            # Probabilistic prediction
            'samples': samples,
            'probabilities': probabilities,
            'dist_stats': dist_stats,
            'directional': directional,
            'confidence': confidence,
            'entropy': entropy,
            
            # Trading signals
            'signals': signals,
            
            # Raw data
            'B_paths': B_paths,
            'K_t': fcd_result['K_t']
        }
    
    def generate_signals(self,
                        fcd_result: Dict,
                        directional: Dict,
                        dist_stats: Dict,
                        coherence: float,
                        tension_norm: float) -> Dict[str, float]:
        """
        Generate trading signals from FCD output using regime-aware permissions
        and the updated raw_score formulation (p_up - p_down).
        """
        regime = fcd_result.get('regime', 'sideways_low_vol')
        permissions = dict(self.regime_permissions.get(regime, {'long': True, 'short': False, 'scale': 1.0}))
        if not self.allow_shorts:
            permissions['short'] = False
        
        p_up = directional.get('upward_prob', 0.0)
        p_down = directional.get('downward_prob', 0.0)
        raw_score = p_up - p_down
        scale = permissions.get('scale', 1.0)
        scaled_score = raw_score * scale
        path_var = fcd_result.get('path_var', None)
        vol_z = fcd_result.get('vol_z', 0.0)
        persistence = fcd_result.get('regime_persistence', 0)
        persistence_history_len = fcd_result.get('regime_history_length', 0)

        # tension-based gating
        interval_label = (self.interval or "1d").lower()
        tension_gate = 0.4 if interval_label == "1d" else 0.45

        vol_regime_label = fcd_result.get('vol_regime', 'mid_vol')
        thresh_scale = 1.0
        if vol_regime_label == 'high_vol':
            thresh_scale = 1.2
        elif vol_regime_label == 'mid_vol':
            thresh_scale = 1.05
        long_component = max(0.0, scaled_score)
        short_component = max(0.0, -scaled_score)
        chop_component = max(0.0, 1.0 - (p_up + p_down))
        total = long_component + short_component + chop_component
        if total > 0:
            long_component /= total
            short_component /= total
            chop_component /= total

        if path_var is not None and len(self.signals_history) >= 5:
            path_var_threshold = 0.020 if interval_label == "1d" else 0.06
            if path_var > path_var_threshold:
                return {
                    'direction': 0,
                    'score': scaled_score,
                    'raw_score': raw_score,
                    'p_up': p_up,
                    'p_down': p_down,
                    'coherence': coherence,
                    'permissions': permissions,
                    'long': long_component,
                    'short': short_component,
                    'chop': chop_component,
                    'net_directional': long_component - short_component,
                    'tension_level': tension_norm,
                    'momentum_bias': directional.get('bias', 0.0),
                    'regime': regime,
                    'reason': 'path_variance_gate',
                    'phase3_gate': True
                }

        if vol_z is None:
            vol_z = 0.0
        vol_gate = 3.0 if interval_label == "1d" else 2.5
        if vol_z > vol_gate:
            return {
                'direction': 0,
                'score': scaled_score,
                'raw_score': raw_score,
                'p_up': p_up,
                'p_down': p_down,
                'coherence': coherence,
                'permissions': permissions,
                'long': long_component,
                'short': short_component,
                'chop': chop_component,
                'net_directional': long_component - short_component,
                'tension_level': tension_norm,
                'momentum_bias': directional.get('bias', 0.0),
                'regime': regime,
                'reason': 'vol_gate',
                'phase3_gate': True
            }

        min_persistence = 2 if interval_label == "1d" else 3
        if persistence_history_len <= 1:
            min_persistence = 1
        if persistence < min_persistence:
            return {
                'direction': 0,
                'score': scaled_score,
                'raw_score': raw_score,
                'p_up': p_up,
                'p_down': p_down,
                'coherence': coherence,
                'permissions': permissions,
                'long': long_component,
                'short': short_component,
                'chop': chop_component,
                'net_directional': long_component - short_component,
                'tension_level': tension_norm,
                'momentum_bias': directional.get('bias', 0.0),
                'regime': regime,
                'reason': 'regime_persistence_gate',
                'phase3_gate': True
            }

        if tension_norm > tension_gate:
            return {
                'direction': 0,
                'score': scaled_score,
                'raw_score': raw_score,
                'p_up': p_up,
                'p_down': p_down,
                'coherence': coherence,
                'permissions': permissions,
                'long': long_component,
                'short': short_component,
                'chop': chop_component,
                'net_directional': long_component - short_component,
                'tension_level': tension_norm,
                'momentum_bias': directional.get('bias', 0.0),
                'regime': regime,
                'reason': 'tension_gate',
                'phase3_gate': True
            }

        effective_long_thresh = self.long_threshold * scale * thresh_scale
        effective_short_thresh = self.short_threshold * scale * thresh_scale

        signal_direction = 0
        if raw_score > effective_long_thresh and permissions.get('long', False):
            signal_direction = 1
        elif raw_score < -effective_short_thresh and permissions.get('short', False):
            signal_direction = -1
        
        return {
            'direction': signal_direction,
            'score': scaled_score,
            'raw_score': raw_score,
            'p_up': p_up,
            'p_down': p_down,
            'coherence': coherence,
            'permissions': permissions,
            'long': long_component,
            'short': short_component,
            'chop': chop_component,
            'net_directional': long_component - short_component,
            'tension_level': tension_norm,
            'momentum_bias': directional.get('bias', 0.0),
            'regime': regime,
            'phase3_gate': False
        }
    
    def compute_multi_scale(self,
                           price_data: Dict[str, np.ndarray],
                           high_data: Dict[str, np.ndarray],
                           low_data: Dict[str, np.ndarray],
                           t_indices: Dict[str, int]) -> Dict:
        """
        Compute multi-scale FCD with coherence metrics.
        
        Parameters:
        -----------
        price_data : Dict[str, np.ndarray]
            Price data for each timeframe
        high_data : Dict[str, np.ndarray]
            High data for each timeframe
        low_data : Dict[str, np.ndarray]
            Low data for each timeframe
        t_indices : Dict[str, int]
            Current time index for each timeframe
            
        Returns:
        --------
        Dict : Multi-scale FCD results
        """
        if not self.enable_multi_scale or self.multi_scale is None:
            raise ValueError("Multi-scale not enabled")
        
        # Compute all timeframes
        results = self.multi_scale.compute_all_timeframes(
            price_data, high_data, low_data, t_indices
        )
        
        # Coherence metrics
        coherence = self.multi_scale.aggregate_coherence(results)
        
        return {
            'timeframe_results': results,
            'coherence': coherence
        }
    
    def reset(self):
        """Reset indicator state."""
        self.fcd.reset()
        self.signals_history = []
        self.c_mag_history = []
        self.volatility_history = []
        self.vol_proxy_history = []
        if self.multi_scale is not None:
            self.multi_scale.reset_all()


def format_output(result: Dict, t: int) -> str:
    """
    Format FCD output for display.
    
    Parameters:
    -----------
    result : Dict
        FCD computation result
    t : int
        Time index
        
    Returns:
    --------
    str : Formatted output
    """
    A_t = result['A_t']
    A_prime = result['A_prime']
    C_mag = result['C_mag']
    signals = result['signals']
    directional = result['directional']
    
    output = f"""
=== FCD-PSE Output at t={t} ===

STATE ESTIMATES:
  A_t (Actual):     [{A_t[0]:.4f}, {A_t[1]:.4f}, {A_t[2]:.4f}]
  A'_t (New):       [{A_prime[0]:.4f}, {A_prime[1]:.4f}, {A_prime[2]:.4f}]

TENSION:
  C_magnitude:      {C_mag:.4f}

DIRECTIONAL BIAS:
  Upward prob:      {directional['upward_prob']:.3f}
  Downward prob:    {directional['downward_prob']:.3f}
  Bias:             {directional['bias']:.3f}

SIGNALS:
  Long:             {signals['long']:.3f}
  Short:            {signals['short']:.3f}
  Chop:             {signals['chop']:.3f}
  Net Directional:  {signals['net_directional']:.3f}
"""
    return output
