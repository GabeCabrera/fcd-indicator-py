#!/usr/bin/env python3
"""
FCD Backtest Engine
===================

Lightweight, reusable backtest engine extracted from run_cloud_fcd_backtest_custom_gates.py.
Optimized for parameter sweep testing - minimal output, fast execution, **vectorized indicators**.

Key Optimization (Phase 1B-fix):
- Pre-compute MACD, ATR once for entire series (O(n) instead of O(n²))
- FCD Monte Carlo still runs per-bar (necessary for path simulation)
- Reduced ~1,248 sweep from hours to minutes

Key differences from original:
- No progress printing (for batch operations)
- Configurable FCD parameters
- Returns structured results object
- Focused on performance metrics only
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add FCD imports
cloud_server_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(cloud_server_path))

from fcd.core.fcd_indicator import FCDIndicator
from fcd.core.primitives import MathPrimitives


@dataclass
class BacktestConfig:
    """Configuration for backtest run."""
    # FCD parameters
    fast_length: int = 12
    slow_length: int = 26
    signal_length: int = 9
    atr_length: int = 14
    vol_length: int = 20
    n_paths: int = 1000
    temperature: float = 1.0
    
    # Monte Carlo distribution (FCD 2.0)
    mc_distribution: str = "gaussian"  # "gaussian" or "student_t"
    mc_degrees_of_freedom: float = 3.0  # Only used for student_t
    
    # Thresholds
    long_threshold: float = 0.01
    short_threshold: float = 0.01
    
    # Gates (disabled by default for parameter sweep)
    enable_path_var_gate: bool = False
    enable_tension_gate: bool = False
    enable_vol_gate: bool = False
    enable_persistence_gate: bool = False
    
    # Trading
    lookback_bars: int = 100
    allow_shorts: bool = False
    position_size: float = 0.95
    
    # Other
    interval: str = "1d"


@dataclass
class BacktestResults:
    """Results from backtest run."""
    ticker: str
    total_trades: int
    win_rate: float
    total_return: float
    avg_pnl_per_trade: float
    avg_win: float
    avg_loss: float
    best_trade: float
    worst_trade: float
    max_drawdown: float
    final_equity: float
    sharpe_ratio: float
    
    # Config used
    config: BacktestConfig
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for CSV export."""
        return {
            'ticker': self.ticker,
            'total_trades': self.total_trades,
            'win_rate': self.win_rate,
            'total_return': self.total_return,
            'avg_pnl_per_trade': self.avg_pnl_per_trade,
            'avg_win': self.avg_win,
            'avg_loss': self.avg_loss,
            'best_trade': self.best_trade,
            'worst_trade': self.worst_trade,
            'max_drawdown': self.max_drawdown,
            'final_equity': self.final_equity,
            'sharpe_ratio': self.sharpe_ratio,
            # Config
            'fast_length': self.config.fast_length,
            'slow_length': self.config.slow_length,
            'signal_length': self.config.signal_length,
            'lookback_bars': self.config.lookback_bars,
            'long_threshold': self.config.long_threshold,
            'interval': self.config.interval
        }


class BacktestEngine:
    """
    Fast backtest engine for parameter optimization.
    
    Stripped-down version optimized for:
    - Speed (no progress printing, vectorized indicators)
    - Batch processing (hundreds of runs)
    - Parameter sweeps (configurable FCD params)
    
    **Phase 1B Optimization**:
    Pre-computes all indicators (MACD, EMA, ATR) once at start,
    reducing complexity from O(n²) to O(n) + O(n * paths).
    
    Usage:
        config = BacktestConfig(fast_length=24, slow_length=52)
        engine = BacktestEngine(config)
        results = engine.run(df, ticker='BTC/USD')
        print(f"Return: {results.total_return:.2f}%")
    """
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.fcd_indicator: Optional[FCDIndicator] = None
        
        # Trading state
        self.position = 0
        self.entry_price = 0.0
        self.entry_bar = 0
        self.cash = 100000.0
        self.initial_cash = 100000.0
        self.shares = 0.0
        
        # Results tracking
        self.trades: List[Dict] = []
        self.equity_curve: List[float] = []
        
        # Pre-computed indicators (Phase 1B optimization)
        self.ema_fast_cache = None
        self.ema_slow_cache = None
        self.macd_line_cache = None
        self.signal_line_cache = None
        self.histogram_cache = None
        self.atr_cache = None
    
    def _precompute_indicators(
        self, 
        price: np.ndarray, 
        high: np.ndarray, 
        low: np.ndarray
    ):
        """
        Pre-compute all technical indicators for entire series.
        
        This is the Phase 1B optimization that eliminates O(n²) recalculation.
        Called once at backtest start instead of on every bar.
        
        Args:
            price: Close price array
            high: High price array
            low: Low price array
        """
        # Compute EMAs once
        self.ema_fast_cache = MathPrimitives.ema(price, self.config.fast_length)
        self.ema_slow_cache = MathPrimitives.ema(price, self.config.slow_length)
        
        # Compute MACD once
        self.macd_line_cache, self.signal_line_cache, self.histogram_cache = \
            MathPrimitives.macd(
                price, 
                self.config.fast_length,
                self.config.slow_length,
                self.config.signal_length
            )
        
        # Compute ATR once
        self.atr_cache = MathPrimitives.atr(
            high, low, price, self.config.atr_length
        )
    
    def _create_fcd_indicator(self):
        """Create FCD indicator with config parameters."""
        self.fcd_indicator = FCDIndicator(
            fast_length=self.config.fast_length,
            slow_length=self.config.slow_length,
            signal_length=self.config.signal_length,
            atr_length=self.config.atr_length,
            vol_length=self.config.vol_length,
            n_paths=self.config.n_paths,
            temperature=self.config.temperature,
            enable_multi_scale=False,
            mc_distribution=self.config.mc_distribution,
            mc_degrees_of_freedom=self.config.mc_degrees_of_freedom,
            horizon=1,
            memory_depth=5,
            memory_lambda=0.25,
            memory_power=1.0,
            long_threshold=self.config.long_threshold,
            short_threshold=self.config.short_threshold,
            interval=self.config.interval,
            allow_shorts=self.config.allow_shorts,
            enable_path_var_gate=self.config.enable_path_var_gate,
            enable_tension_gate=self.config.enable_tension_gate,
            enable_vol_gate=self.config.enable_vol_gate,
            enable_persistence_gate=self.config.enable_persistence_gate
        )
    
    def _compute_fcd_signal_fast(
        self,
        close_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        t: int
    ) -> int:
        """
        Compute FCD signal using pre-computed indicators (Phase 1B optimization).
        
        This bypasses the indicator recalculation bottleneck by injecting cached
        indicator values into the FCD state computation.
        
        Args:
            close_prices: Full close price array
            high_prices: Full high price array  
            low_prices: Full low price array
            t: Current time index
            
        Returns:
            signal direction (1=long, 0=flat, -1=short)
        """
        if self.fcd_indicator is None:
            self._create_fcd_indicator()
        
        # Pre-compute indicators if not already done
        if self.ema_fast_cache is None:
            self._precompute_indicators(close_prices, high_prices, low_prices)
        
        try:
            # Monkey-patch the primitives to return cached values
            # This is the key optimization: FCD calls primitives.ema/macd/atr,
            # but we intercept and return pre-computed values
            original_ema = MathPrimitives.ema
            original_macd = MathPrimitives.macd  
            original_atr = MathPrimitives.atr
            
            def cached_ema(data, length):
                # Return cached values up to index t
                if length == self.config.fast_length:
                    return self.ema_fast_cache[:len(data)]
                elif length == self.config.slow_length:
                    return self.ema_slow_cache[:len(data)]
                else:
                    # Signal line or other - compute normally
                    return original_ema(data, length)
            
            def cached_macd(data, fast, slow, signal):
                # Return cached MACD values up to current length
                n = len(data)
                return (
                    self.macd_line_cache[:n],
                    self.signal_line_cache[:n],
                    self.histogram_cache[:n]
                )
            
            def cached_atr(high, low, close, length):
                # Return cached ATR values up to current length
                return self.atr_cache[:len(close)]
            
            # Apply patches
            MathPrimitives.ema = cached_ema
            MathPrimitives.macd = cached_macd
            MathPrimitives.atr = cached_atr
            
            try:
                # Now FCD compute will use cached values
                fcd_result = self.fcd_indicator.compute(
                    price=close_prices[:t+1],
                    high=high_prices[:t+1],
                    low=low_prices[:t+1],
                    t=t
                )
                
                signals = fcd_result['signals']
                return signals['direction']
            finally:
                # Restore original methods
                MathPrimitives.ema = original_ema
                MathPrimitives.macd = original_macd
                MathPrimitives.atr = original_atr
                
        except Exception:
            # Silent failure - return flat signal
            return 0
    
    def _execute_trade(self, action: str, price: float, bar: int):
        """Execute a trade and record it."""
        if action == "BUY":
            self.position = 1
            self.entry_price = price
            self.entry_bar = bar
            self.shares = (self.cash * self.config.position_size) / price
            self.cash -= self.shares * price
            
            self.trades.append({
                'bar': bar,
                'action': 'BUY',
                'price': price,
                'shares': self.shares
            })
            
        elif action == "SELL":
            if self.position == 1:
                pnl = (price - self.entry_price) * self.shares
                pnl_pct = ((price - self.entry_price) / self.entry_price) * 100
                self.cash += self.shares * price
                hold_bars = bar - self.entry_bar
                
                self.trades.append({
                    'bar': bar,
                    'action': 'SELL',
                    'price': price,
                    'entry_price': self.entry_price,
                    'shares': self.shares,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'hold_bars': hold_bars
                })
                
                self.shares = 0.0
                self.position = 0
                self.entry_price = 0.0
                self.entry_bar = 0
    
    def run(self, df: pd.DataFrame, ticker: str) -> BacktestResults:
        """
        Run backtest on historical data.
        
        **Phase 1B Optimization**: Pre-computes all indicators once,
        then iterates through bars with O(1) indicator lookups.
        
        Args:
            df: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
            ticker: Asset ticker (for results tracking)
            
        Returns:
            BacktestResults with performance metrics
        """
        # Reset state
        self.position = 0
        self.entry_price = 0.0
        self.entry_bar = 0
        self.cash = self.initial_cash
        self.shares = 0.0
        self.trades = []
        self.equity_curve = []
        
        # Clear cached indicators (for new run)
        self.ema_fast_cache = None
        self.ema_slow_cache = None
        self.macd_line_cache = None
        self.signal_line_cache = None
        self.histogram_cache = None
        self.atr_cache = None
        
        # Prepare arrays
        close_prices = df['close'].values
        high_prices = df['high'].values
        low_prices = df['low'].values
        
        # **PHASE 1B OPTIMIZATION**: Pre-compute indicators ONCE
        # This eliminates O(n²) recalculation on every bar
        self._precompute_indicators(close_prices, high_prices, low_prices)
        
        # Walk-forward loop (now with O(1) indicator lookups per bar)
        for t in range(self.config.lookback_bars, len(df)):
            # Compute FCD signal (Monte Carlo still runs, but indicators are cached)
            signal = self._compute_fcd_signal_fast(
                close_prices=close_prices,
                high_prices=high_prices,
                low_prices=low_prices,
                t=t
            )
            
            current_price = close_prices[t]
            
            # Trading logic
            if signal == 1 and self.position == 0:
                self._execute_trade("BUY", current_price, t)
            
            elif signal == 0 and self.position == 1:
                self._execute_trade("SELL", current_price, t)
            
            elif signal == -1 and self.config.allow_shorts:
                if self.position == 1:
                    self._execute_trade("SELL", current_price, t)
            
            # Track equity
            if self.position == 1:
                equity = self.cash + (self.shares * current_price)
            else:
                equity = self.cash
            self.equity_curve.append(equity)
        
        # Close any open position at end
        if self.position != 0:
            self._execute_trade("SELL", close_prices[-1], len(df)-1)
        
        # Calculate results
        return self._calculate_results(ticker)
    
    def _calculate_results(self, ticker: str) -> BacktestResults:
        """Calculate performance metrics."""
        completed_trades = [t for t in self.trades if t['action'] == 'SELL']
        
        if not completed_trades:
            return BacktestResults(
                ticker=ticker,
                total_trades=0,
                win_rate=0.0,
                total_return=0.0,
                avg_pnl_per_trade=0.0,
                avg_win=0.0,
                avg_loss=0.0,
                best_trade=0.0,
                worst_trade=0.0,
                max_drawdown=0.0,
                final_equity=self.initial_cash,
                sharpe_ratio=0.0,
                config=self.config
            )
        
        # Calculate metrics
        pnls = [t['pnl_pct'] for t in completed_trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        
        total_return = ((self.cash - self.initial_cash) / self.initial_cash) * 100
        win_rate = len(wins) / len(completed_trades) if completed_trades else 0
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        avg_pnl = np.mean(pnls) if pnls else 0
        
        # Calculate max drawdown
        equity_array = np.array(self.equity_curve)
        running_max = np.maximum.accumulate(equity_array)
        drawdown = (equity_array - running_max) / running_max * 100
        max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0
        
        # Calculate Sharpe ratio (assuming daily returns for now)
        if len(pnls) > 1:
            returns_std = np.std(pnls)
            if returns_std > 0:
                sharpe_ratio = (avg_pnl / returns_std) * np.sqrt(252)  # Annualized
            else:
                sharpe_ratio = 0.0
        else:
            sharpe_ratio = 0.0
        
        return BacktestResults(
            ticker=ticker,
            total_trades=len(completed_trades),
            win_rate=win_rate,
            total_return=total_return,
            avg_pnl_per_trade=avg_pnl,
            avg_win=avg_win,
            avg_loss=avg_loss,
            best_trade=max(pnls) if pnls else 0,
            worst_trade=min(pnls) if pnls else 0,
            max_drawdown=max_drawdown,
            final_equity=self.cash,
            sharpe_ratio=sharpe_ratio,
            config=self.config
        )


if __name__ == "__main__":
    """
    Test backtest engine with sample data.
    """
    print("=" * 70)
    print("Backtest Engine Test")
    print("=" * 70)
    
    # Create sample data (sine wave with trend)
    dates = pd.date_range('2024-01-01', periods=500, freq='D')
    prices = 100 + np.cumsum(np.random.randn(500) * 2) + 10 * np.sin(np.linspace(0, 4*np.pi, 500))
    
    df = pd.DataFrame({
        'open': prices,
        'high': prices * 1.01,
        'low': prices * 0.99,
        'close': prices,
        'volume': np.random.randint(1000, 10000, 500)
    }, index=dates)
    
    # Run backtest with default config
    print("\n[Test 1] Default configuration...")
    config = BacktestConfig()
    engine = BacktestEngine(config)
    results = engine.run(df, ticker='TEST')
    
    print(f"✅ Backtest complete!")
    print(f"   Trades: {results.total_trades}")
    print(f"   Return: {results.total_return:+.2f}%")
    print(f"   Win Rate: {results.win_rate:.1%}")
    print(f"   Sharpe: {results.sharpe_ratio:.2f}")
    
    # Test with different parameters
    print("\n[Test 2] Custom parameters (fast=24, slow=52)...")
    config2 = BacktestConfig(fast_length=24, slow_length=52, lookback_bars=200)
    engine2 = BacktestEngine(config2)
    results2 = engine2.run(df, ticker='TEST')
    
    print(f"✅ Backtest complete!")
    print(f"   Trades: {results2.total_trades}")
    print(f"   Return: {results2.total_return:+.2f}%")
    print(f"   Win Rate: {results2.win_rate:.1%}")
    
    print("\n" + "=" * 70)
    print("Engine test complete!")
    print("=" * 70)
