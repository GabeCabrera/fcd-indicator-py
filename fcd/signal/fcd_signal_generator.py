"""
FCD Signal Generator - Cloud Server Edition
============================================
Simplified signal generator for cloud deployment that integrates
with the full FCD-PSE model.

IMPORTANT: This preserves the complete FCD core logic without modification.
"""

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple
import pandas as pd
import numpy as np

# Import FCD core components (DO NOT MODIFY)
from fcd.core.fcd_indicator import FCDIndicator


class FCDSignalGenerator:
    """
    Cloud-ready FCD signal generator.
    
    Features:
    - Real-time bar data caching
    - BecomingScore filtering
    - Complete FCD-PSE integration
    - Configurable thresholds
    """
    
    def __init__(
        self,
        min_becoming_score: float = 0.0,
        lookback_bars: int = 100,
        fcd_long_threshold: float = 0.1,
        fcd_short_threshold: float = 0.1,
        allow_shorts: bool = False,
        interval: str = "1d"
    ):
        """
        Initialize FCD signal generator.
        
        Args:
            min_becoming_score: Minimum BecomingScore to trade (default 0.0 = no filter)
            lookback_bars: Number of bars for FCD calculation (default 100)
            fcd_long_threshold: Threshold for LONG signals (default 0.1)
            fcd_short_threshold: Threshold for SHORT signals (default 0.1)
            allow_shorts: Whether to allow short signals (default False = long-only)
            interval: Data interval (e.g., "1d", "1h") for FCD configuration
        """
        self.min_becoming_score = min_becoming_score
        self.lookback_bars = lookback_bars
        self.fcd_long_threshold = fcd_long_threshold
        self.fcd_short_threshold = fcd_short_threshold
        self.allow_shorts = allow_shorts
        self.interval = interval
        
        # Bar cache per symbol
        self.bar_cache: Dict[str, pd.DataFrame] = {}
        
        # BecomingScore rankings
        self.becoming_scores: Dict[str, float] = {}
        
        # FCD indicator instances per symbol
        self.fcd_indicators: Dict[str, FCDIndicator] = {}
        
        print(f"[FCD] Signal Generator Initialized")
        print(f"  Lookback Bars: {lookback_bars}")
        print(f"  Long Threshold: {fcd_long_threshold}")
        print(f"  Short Threshold: {fcd_short_threshold}")
        print(f"  Allow Shorts: {allow_shorts}")
        print(f"  Interval: {interval}")
        sys.stdout.flush()
    
    def load_becoming_scores(self, rankings_dir: str) -> None:
        """
        Load BecomingScore rankings from CSV files.
        
        Args:
            rankings_dir: Directory containing ranking CSV files
        """
        rankings_path = Path(rankings_dir)
        
        if not rankings_path.exists():
            print(f"[FCD] Rankings directory not found: {rankings_dir}")
            return
        
        # Look for consolidated futures rankings
        consolidated_file = rankings_path / "consolidated_futures.csv"
        
        if consolidated_file.exists():
            try:
                df = pd.read_csv(consolidated_file)
                
                for _, row in df.iterrows():
                    ticker = row['ticker']
                    score = row['becoming_score_mean']
                    
                    if pd.notna(score):
                        self.becoming_scores[ticker] = float(score)
                
                print(f"[FCD] Loaded BecomingScores for {len(self.becoming_scores)} instruments")
                
                # Show tradeable instruments
                tradeable = [t for t, s in self.becoming_scores.items() if s >= self.min_becoming_score]
                print(f"[FCD] Tradeable instruments (score >= {self.min_becoming_score}): {len(tradeable)}")
                
            except Exception as e:
                print(f"[FCD] Error loading rankings: {e}")
        else:
            print(f"[FCD] Consolidated rankings not found: {consolidated_file}")
        
        sys.stdout.flush()
    
    def _get_or_create_indicator(self, symbol: str) -> FCDIndicator:
        """Get existing or create new FCD indicator for symbol"""
        if symbol not in self.fcd_indicators:
            self.fcd_indicators[symbol] = FCDIndicator(
                fast_length=12,
                slow_length=26,
                signal_length=9,
                atr_length=14,
                vol_length=20,
                n_paths=1000,
                temperature=1.0,
                enable_multi_scale=False,
                horizon=3,
                memory_depth=5,
                memory_lambda=0.25,
                memory_power=1.0,
                long_threshold=self.fcd_long_threshold,
                short_threshold=self.fcd_short_threshold,
                interval=self.interval,
                allow_shorts=self.allow_shorts
            )
        
        return self.fcd_indicators[symbol]
    
    def update_bar_data(self, symbol: str, bar_data: Dict) -> None:
        """
        Update bar cache with new bar.
        
        Args:
            symbol: Trading symbol
            bar_data: Dict with timestamp, open, high, low, close, volume
        """
        if symbol not in self.bar_cache:
            self.bar_cache[symbol] = pd.DataFrame()
        
        # Create new bar
        new_bar = pd.DataFrame([{
            'timestamp': pd.to_datetime(bar_data['timestamp'], unit='ms'),
            'open': bar_data['open'],
            'high': bar_data['high'],
            'low': bar_data['low'],
            'close': bar_data['close'],
            'volume': bar_data['volume']
        }])
        
        # Append to cache
        self.bar_cache[symbol] = pd.concat(
            [self.bar_cache[symbol], new_bar],
            ignore_index=True
        )
        
        # Keep only last N bars
        if len(self.bar_cache[symbol]) > self.lookback_bars * 2:
            self.bar_cache[symbol] = self.bar_cache[symbol].iloc[-self.lookback_bars:]
    
    def get_signal(
        self,
        symbol: str,
        bar_data: Dict,
        current_position: str = "FLAT"
    ) -> Tuple[str, Optional[Dict]]:
        """
        Generate FCD trading signal.
        
        Args:
            symbol: Trading symbol
            bar_data: Current bar data
            current_position: "LONG" or "FLAT"
        
        Returns:
            Tuple of (signal, metadata):
            - signal: "LONG", "FLAT", or "HOLD"
            - metadata: Dict with FCD values and diagnostics
        """
        # Update bar cache
        self.update_bar_data(symbol, bar_data)
        
        # Check minimum bars
        if symbol not in self.bar_cache or len(self.bar_cache[symbol]) < 30:
            return "HOLD", {
                "reason": "insufficient_data",
                "bars_available": len(self.bar_cache.get(symbol, [])),
                "bars_required": 30
            }
        
        # Check BecomingScore filter
        becoming_score = self.becoming_scores.get(symbol, 0.0)
        if self.min_becoming_score > 0 and becoming_score < self.min_becoming_score:
            return "FLAT", {
                "reason": "low_becoming_score",
                "becoming_score": becoming_score,
                "threshold": self.min_becoming_score
            }
        
        # Calculate FCD indicator
        try:
            df = self.bar_cache[symbol].copy()
            
            # Prepare arrays for FCD
            close_prices = df['close'].values
            high_prices = df['high'].values
            low_prices = df['low'].values
            
            t = len(close_prices) - 1  # Current time index
            
            # Get or create FCD indicator for this symbol
            fcd = self._get_or_create_indicator(symbol)
            
            # Compute FCD (CORE LOGIC - DO NOT MODIFY)
            fcd_result = fcd.compute(
                price=close_prices,
                high=high_prices,
                low=low_prices,
                t=t
            )
            
            # Extract signal information
            signals = fcd_result['signals']
            signal_direction = signals['direction']  # 1=LONG, -1=SHORT, 0=FLAT
            
            # Get FCD metrics
            fcd_value = fcd_result.get('normalized_coherence', 0.0)
            coherence = fcd_result.get('coherence', 0.0)
            raw_score = signals.get('raw_score', 0.0)
            confidence = signals.get('p_up', 0.0) - signals.get('p_down', 0.0)
            
            # Determine trading signal
            if signal_direction == 1:
                signal = "LONG"
                reason = "fcd_long_signal"
            elif signal_direction == -1 and self.allow_shorts:
                signal = "FLAT"  # Close position (or SHORT if implementing shorts)
                reason = "fcd_short_signal"
            else:
                signal = "FLAT"
                reason = "fcd_neutral"
            
            # Avoid redundant signals
            if (signal == "LONG" and current_position == "LONG") or \
               (signal == "FLAT" and current_position == "FLAT"):
                signal = "HOLD"
                reason = "position_unchanged"
            
            # Build metadata
            metadata = {
                "fcd_value": float(fcd_value),
                "coherence": float(coherence),
                "raw_score": float(raw_score),
                "confidence": float(confidence),
                "becoming_score": becoming_score,
                "signal_direction": signal_direction,
                "reason": reason,
                "long_component": signals.get('long', 0.0),
                "short_component": signals.get('short', 0.0),
                "chop_component": signals.get('chop', 0.0),
                "tension_level": signals.get('tension_level', 0.0),
                "regime": fcd_result.get('regime', 'unknown'),
                "bars_used": len(close_prices)
            }
            
            return signal, metadata
            
        except Exception as e:
            print(f"[FCD] Error computing signal for {symbol}: {e}")
            import traceback
            traceback.print_exc()
            sys.stdout.flush()
            
            return "HOLD", {
                "reason": "calculation_error",
                "error": str(e)
            }
    
    def get_tradeable_instruments(self) -> list:
        """Get list of tradeable instruments based on BecomingScore filter"""
        if not self.becoming_scores:
            return []
        
        tradeable = [
            ticker for ticker, score in self.becoming_scores.items()
            if score >= self.min_becoming_score
        ]
        
        return sorted(tradeable, key=lambda t: self.becoming_scores[t], reverse=True)


if __name__ == "__main__":
    # Test signal generator
    print("\n=== Testing FCD Signal Generator ===\n")
    
    generator = FCDSignalGenerator(
        min_becoming_score=0.0,
        lookback_bars=100,
        fcd_long_threshold=0.1
    )
    
    # Load rankings
    rankings_dir = Path(__file__).parent.parent / "fcd" / "rankings"
    if rankings_dir.exists():
        generator.load_becoming_scores(str(rankings_dir))
    
    # Simulate bar data
    print("\n=== Simulating Bars ===\n")
    
    import random
    base_price = 2450.0
    
    for i in range(50):
        # Simulate random walk
        change = random.uniform(-5, 5)
        base_price += change
        
        bar_data = {
            'timestamp': 1700000000000 + (i * 86400000),  # Daily bars
            'open': base_price,
            'high': base_price + random.uniform(0, 3),
            'low': base_price - random.uniform(0, 3),
            'close': base_price + random.uniform(-2, 2),
            'volume': random.uniform(1000, 5000)
        }
        
        signal, metadata = generator.get_signal('MGC=F', bar_data, 'FLAT')
        
        if i >= 30 and signal != "HOLD":  # After warmup
            print(f"\nBar {i+1}:")
            print(f"  Signal: {signal}")
            print(f"  FCD: {metadata.get('fcd_value', 'N/A'):.4f}")
            print(f"  Reason: {metadata.get('reason', 'N/A')}")
    
    print("\n✅ Test complete")
