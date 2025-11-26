"""
FCD-PSE Multi-Scale System
===========================
Cross-timeframe FCD analysis with coherence metrics.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from .fcd_state import FCDState
from .probabilistic import ProbabilisticPredictor


class MultiScaleFCD:
    """
    Multi-timeframe FCD indicator system.
    
    Computes independent FCD transformations across multiple timeframes
    and measures cross-scale coherence.
    """
    
    def __init__(self,
                 timeframes: Optional[List[str]] = None,
                 timeframe_configs: Optional[Dict[str, Dict]] = None,
                 weights: Optional[Dict[str, float]] = None):
        """
        Initialize multi-scale FCD system.
        
        Parameters:
        -----------
        timeframes : List[str], optional
            List of timeframe labels (e.g., ['1m', '5m', '15m', '1h', '4h', '1d'])
        timeframe_configs : Dict[str, Dict], optional
            Configuration for each timeframe (EMA lengths, etc.)
        weights : Dict[str, float], optional
            Weights for each timeframe in coherence calculation
        """
        if timeframes is None:
            timeframes = ['5m', '15m', '1h', '4h']
        
        self.timeframes = timeframes
        
        # Initialize FCD state for each timeframe
        self.fcd_systems = {}
        
        for tf in timeframes:
            if timeframe_configs and tf in timeframe_configs:
                config = timeframe_configs[tf]
            else:
                # Default configuration
                config = self._default_config(tf)
            
            self.fcd_systems[tf] = FCDState(**config)
        
        # Coherence weights
        if weights is None:
            # Default: higher timeframes get more weight
            weights = self._default_weights(timeframes)
        self.weights = weights
        
        # Initialize predictor
        self.predictor = ProbabilisticPredictor()
    
    def _default_config(self, timeframe: str) -> Dict:
        """Generate default configuration for timeframe."""
        # Scale parameters based on timeframe
        tf_scales = {
            '1m': 1.0,
            '5m': 1.5,
            '15m': 2.0,
            '1h': 3.0,
            '4h': 4.0,
            '1d': 5.0
        }
        
        scale = tf_scales.get(timeframe, 2.0)
        
        return {
            'fast_length': int(12 * scale),
            'slow_length': int(26 * scale),
            'signal_length': int(9 * scale),
            'atr_length': int(14 * scale),
            'vol_length': int(20 * scale),
            'n_paths': 1000,
            'obs_noise_scale': 0.1,
            'diffusion_scale': 1.0
        }
    
    def _default_weights(self, timeframes: List[str]) -> Dict[str, float]:
        """Generate default weights for timeframes."""
        tf_hierarchy = {
            '1m': 0.5,
            '5m': 1.0,
            '15m': 1.5,
            '1h': 2.0,
            '4h': 2.5,
            '1d': 3.0
        }
        
        weights = {}
        total_weight = 0.0
        
        for tf in timeframes:
            w = tf_hierarchy.get(tf, 1.0)
            weights[tf] = w
            total_weight += w
        
        # Normalize
        for tf in timeframes:
            weights[tf] /= total_weight
        
        return weights
    
    def compute_all_timeframes(self,
                              price_data: Dict[str, np.ndarray],
                              high_data: Dict[str, np.ndarray],
                              low_data: Dict[str, np.ndarray],
                              t_indices: Dict[str, int]) -> Dict[str, Dict]:
        """
        Compute FCD for all timeframes.
        
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
        Dict[str, Dict] : FCD results for each timeframe
        """
        results = {}
        
        for tf in self.timeframes:
            if tf not in price_data:
                continue
            
            fcd_result = self.fcd_systems[tf].full_fcd_cycle(
                price=price_data[tf],
                high=high_data[tf],
                low=low_data[tf],
                t=t_indices[tf]
            )
            
            results[tf] = fcd_result
        
        return results
    
    def coherence_metric_tension(self, 
                                results: Dict[str, Dict]) -> Dict[str, float]:
        """
        Compute cross-scale coherence based on tension (C_t) alignment.
        
        Parameters:
        -----------
        results : Dict[str, Dict]
            FCD results for all timeframes
            
        Returns:
        --------
        Dict containing:
            - align_C: Weighted sum of tension signs
            - tension_coherence: Normalized coherence metric
        """
        weighted_signs = []
        
        for tf in self.timeframes:
            if tf not in results:
                continue
            
            C_t = results[tf]['C_t']
            momentum_tension = C_t[1]  # Momentum component
            
            sign_value = np.sign(momentum_tension)
            weight = self.weights[tf]
            
            weighted_signs.append(sign_value * weight)
        
        align_C = np.sum(weighted_signs)
        
        # Normalize to [-1, 1]
        max_align = np.sum([self.weights[tf] for tf in self.timeframes if tf in results])
        tension_coherence = align_C / max_align if max_align > 0 else 0.0
        
        return {
            'align_C': align_C,
            'tension_coherence': tension_coherence
        }
    
    def coherence_metric_state(self,
                              results: Dict[str, Dict]) -> Dict[str, float]:
        """
        Compute cross-scale coherence based on A'_t alignment.
        
        Parameters:
        -----------
        results : Dict[str, Dict]
            FCD results for all timeframes
            
        Returns:
        --------
        Dict containing:
            - align_A: Weighted sum of state momentum signs
            - state_coherence: Normalized coherence metric
        """
        weighted_signs = []
        
        for tf in self.timeframes:
            if tf not in results:
                continue
            
            A_prime = results[tf]['A_prime']
            momentum_state = A_prime[1]  # Momentum component
            
            sign_value = np.sign(momentum_state)
            weight = self.weights[tf]
            
            weighted_signs.append(sign_value * weight)
        
        align_A = np.sum(weighted_signs)
        
        # Normalize to [-1, 1]
        max_align = np.sum([self.weights[tf] for tf in self.timeframes if tf in results])
        state_coherence = align_A / max_align if max_align > 0 else 0.0
        
        return {
            'align_A': align_A,
            'state_coherence': state_coherence
        }
    
    def coherence_metric_trend(self,
                              results: Dict[str, Dict]) -> Dict[str, float]:
        """
        Compute trend alignment across timeframes.
        
        Parameters:
        -----------
        results : Dict[str, Dict]
            FCD results for all timeframes
            
        Returns:
        --------
        Dict containing:
            - trend_coherence: Trend alignment metric
            - trend_direction: Overall trend direction (-1, 0, 1)
        """
        weighted_trends = []
        
        for tf in self.timeframes:
            if tf not in results:
                continue
            
            A_prime = results[tf]['A_prime']
            trend_value = A_prime[0]  # Trend component
            
            weight = self.weights[tf]
            weighted_trends.append(trend_value * weight)
        
        # Mean weighted trend
        mean_trend = np.mean(weighted_trends) if weighted_trends else 0.0
        
        # Coherence: how consistent are the trends?
        if len(weighted_trends) > 1:
            trend_std = np.std(weighted_trends)
            trend_mean_abs = np.abs(mean_trend)
            
            # Coherence is high when std is low relative to mean
            coherence = trend_mean_abs / (trend_std + 1e-6) if trend_mean_abs > 0 else 0.0
            coherence = np.tanh(coherence)  # Normalize to [0, 1)
        else:
            coherence = 1.0 if len(weighted_trends) == 1 else 0.0
        
        trend_direction = np.sign(mean_trend)
        
        return {
            'trend_coherence': coherence,
            'trend_direction': trend_direction,
            'mean_trend': mean_trend
        }
    
    def volatility_regime(self, results: Dict[str, Dict]) -> Dict[str, float]:
        """
        Determine volatility regime across timeframes.
        
        Parameters:
        -----------
        results : Dict[str, Dict]
            FCD results for all timeframes
            
        Returns:
        --------
        Dict containing:
            - mean_volatility: Weighted mean volatility
            - vol_regime: 'low', 'medium', 'high'
        """
        weighted_vols = []
        
        for tf in self.timeframes:
            if tf not in results:
                continue
            
            A_t = results[tf]['A_t']
            vol_value = A_t[2]  # Volatility component
            
            weight = self.weights[tf]
            weighted_vols.append(vol_value * weight)
        
        mean_vol = np.mean(weighted_vols) if weighted_vols else 0.0
        
        # Classify regime
        if mean_vol < 0.5:
            regime = 'low'
        elif mean_vol < 1.5:
            regime = 'medium'
        else:
            regime = 'high'
        
        return {
            'mean_volatility': mean_vol,
            'vol_regime': regime
        }
    
    def aggregate_coherence(self, results: Dict[str, Dict]) -> Dict[str, float]:
        """
        Compute all coherence metrics.
        
        Parameters:
        -----------
        results : Dict[str, Dict]
            FCD results for all timeframes
            
        Returns:
        --------
        Dict : All coherence metrics
        """
        tension_metrics = self.coherence_metric_tension(results)
        state_metrics = self.coherence_metric_state(results)
        trend_metrics = self.coherence_metric_trend(results)
        vol_metrics = self.volatility_regime(results)
        
        # Overall coherence score
        overall_coherence = (
            0.3 * tension_metrics['tension_coherence'] +
            0.3 * state_metrics['state_coherence'] +
            0.4 * trend_metrics['trend_coherence']
        )
        
        return {
            **tension_metrics,
            **state_metrics,
            **trend_metrics,
            **vol_metrics,
            'overall_coherence': overall_coherence
        }
    
    def reset_all(self):
        """Reset all timeframe FCD systems."""
        for fcd_system in self.fcd_systems.values():
            fcd_system.reset()
