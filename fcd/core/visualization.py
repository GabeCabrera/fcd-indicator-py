"""
FCD-PSE Visualization Tools
============================
Plotting and visualization utilities for FCD indicator.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import Dict, List, Optional
import matplotlib.patches as mpatches


class FCDVisualizer:
    """Visualization tools for FCD-PSE indicator."""
    
    def __init__(self, figsize: tuple = (16, 12)):
        """
        Initialize visualizer.
        
        Parameters:
        -----------
        figsize : tuple
            Figure size (width, height)
        """
        self.figsize = figsize
    
    def plot_full_analysis(self,
                          price: np.ndarray,
                          results_history: List[Dict],
                          start_idx: int = 0) -> plt.Figure:
        """
        Create comprehensive FCD analysis plot.
        
        Parameters:
        -----------
        price : np.ndarray
            Price series
        results_history : List[Dict]
            History of FCD results
        start_idx : int
            Starting index for results
            
        Returns:
        --------
        plt.Figure : Matplotlib figure
        """
        fig = plt.figure(figsize=self.figsize)
        gs = GridSpec(6, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        n_points = len(results_history)
        time_axis = np.arange(start_idx, start_idx + n_points)
        
        # Extract time series
        A_trend = np.array([r['A_t'][0] for r in results_history])
        A_mom = np.array([r['A_t'][1] for r in results_history])
        A_vol = np.array([r['A_t'][2] for r in results_history])
        
        A_prime_trend = np.array([r['A_prime'][0] for r in results_history])
        A_prime_mom = np.array([r['A_prime'][1] for r in results_history])
        
        C_mag = np.array([r['C_mag'] for r in results_history])
        
        long_signal = np.array([r['signals']['long'] for r in results_history])
        short_signal = np.array([r['signals']['short'] for r in results_history])
        chop_signal = np.array([r['signals']['chop'] for r in results_history])
        
        upward_prob = np.array([r['directional']['upward_prob'] for r in results_history])
        downward_prob = np.array([r['directional']['downward_prob'] for r in results_history])
        
        # 1. Price with trend overlay
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(time_axis, price[start_idx:start_idx+n_points], 'k-', label='Price', linewidth=1.5)
        ax1_twin = ax1.twinx()
        ax1_twin.plot(time_axis, A_trend, 'b-', label='A_t Trend', alpha=0.7)
        ax1_twin.plot(time_axis, A_prime_trend, 'r--', label="A'_t Trend", alpha=0.7)
        ax1.set_ylabel('Price', fontsize=10)
        ax1_twin.set_ylabel('Trend State', fontsize=10)
        ax1.set_title('Price and Trend State Evolution', fontsize=12, fontweight='bold')
        ax1.legend(loc='upper left', fontsize=8)
        ax1_twin.legend(loc='upper right', fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # 2. Momentum states
        ax2 = fig.add_subplot(gs[1, :])
        ax2.plot(time_axis, A_mom, 'g-', label='A_t Momentum', linewidth=1.5)
        ax2.plot(time_axis, A_prime_mom, 'm--', label="A'_t Momentum", linewidth=1.5)
        ax2.axhline(y=0, color='k', linestyle=':', alpha=0.5)
        ax2.fill_between(time_axis, 0, A_mom, where=(A_mom > 0), alpha=0.3, color='green', label='Positive')
        ax2.fill_between(time_axis, 0, A_mom, where=(A_mom < 0), alpha=0.3, color='red', label='Negative')
        ax2.set_ylabel('Momentum', fontsize=10)
        ax2.set_title('Momentum State Evolution', fontsize=12, fontweight='bold')
        ax2.legend(loc='best', fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        # 3. Tension magnitude
        ax3 = fig.add_subplot(gs[2, 0])
        ax3.plot(time_axis, C_mag, 'r-', linewidth=2)
        ax3.fill_between(time_axis, 0, C_mag, alpha=0.3, color='red')
        ax3.set_ylabel('Tension', fontsize=10)
        ax3.set_title('Internal Resolution (C_mag)', fontsize=11, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # 4. Volatility
        ax4 = fig.add_subplot(gs[2, 1])
        ax4.plot(time_axis, A_vol, 'orange', linewidth=2)
        ax4.fill_between(time_axis, 0, A_vol, alpha=0.3, color='orange')
        ax4.set_ylabel('Volatility', fontsize=10)
        ax4.set_title('Volatility State (A_t Vol)', fontsize=11, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # 5. Directional probabilities
        ax5 = fig.add_subplot(gs[3, :])
        ax5.plot(time_axis, upward_prob, 'g-', label='Upward Prob', linewidth=2)
        ax5.plot(time_axis, downward_prob, 'r-', label='Downward Prob', linewidth=2)
        ax5.axhline(y=0.5, color='k', linestyle='--', alpha=0.5)
        ax5.fill_between(time_axis, upward_prob, downward_prob, 
                        where=(upward_prob > downward_prob), alpha=0.3, color='green')
        ax5.fill_between(time_axis, upward_prob, downward_prob, 
                        where=(downward_prob > upward_prob), alpha=0.3, color='red')
        ax5.set_ylabel('Probability', fontsize=10)
        ax5.set_title('Directional Probabilities', fontsize=12, fontweight='bold')
        ax5.legend(loc='best', fontsize=8)
        ax5.grid(True, alpha=0.3)
        ax5.set_ylim([0, 1])
        
        # 6. Trading signals
        ax6 = fig.add_subplot(gs[4, :])
        width = 0.8
        ax6.bar(time_axis, long_signal, width, label='Long', color='green', alpha=0.7)
        ax6.bar(time_axis, -short_signal, width, label='Short', color='red', alpha=0.7)
        ax6.bar(time_axis, chop_signal * 0.5, width, label='Chop', color='gray', alpha=0.5)
        ax6.axhline(y=0, color='k', linestyle='-', linewidth=1)
        ax6.set_ylabel('Signal Strength', fontsize=10)
        ax6.set_title('Trading Signals', fontsize=12, fontweight='bold')
        ax6.legend(loc='best', fontsize=8)
        ax6.grid(True, alpha=0.3)
        
        # 7. Distribution visualization (latest)
        if len(results_history) > 0:
            latest = results_history[-1]
            samples = latest['samples']
            probabilities = latest['probabilities']
            
            # Momentum distribution
            ax7 = fig.add_subplot(gs[5, 0])
            mom_samples = samples[:, 1]
            ax7.hist(mom_samples, bins=50, weights=probabilities, alpha=0.7, color='blue', edgecolor='black')
            ax7.axvline(x=A_prime_mom[-1], color='r', linestyle='--', linewidth=2, label="A'_t Momentum")
            ax7.set_xlabel('Momentum Value', fontsize=10)
            ax7.set_ylabel('Probability Density', fontsize=10)
            ax7.set_title('Latest Momentum Distribution', fontsize=11, fontweight='bold')
            ax7.legend(fontsize=8)
            ax7.grid(True, alpha=0.3)
            
            # Volatility distribution
            ax8 = fig.add_subplot(gs[5, 1])
            vol_samples = samples[:, 2]
            ax8.hist(vol_samples, bins=50, weights=probabilities, alpha=0.7, color='orange', edgecolor='black')
            ax8.axvline(x=A_vol[-1], color='r', linestyle='--', linewidth=2, label='A_t Volatility')
            ax8.set_xlabel('Volatility Value', fontsize=10)
            ax8.set_ylabel('Probability Density', fontsize=10)
            ax8.set_title('Latest Volatility Distribution', fontsize=11, fontweight='bold')
            ax8.legend(fontsize=8)
            ax8.grid(True, alpha=0.3)
        
        plt.suptitle('FCD-PSE Indicator Complete Analysis', fontsize=14, fontweight='bold', y=0.995)
        
        return fig
    
    def plot_state_space(self, results_history: List[Dict]) -> plt.Figure:
        """
        Plot state space trajectory (3D or 2D projections).
        
        Parameters:
        -----------
        results_history : List[Dict]
            History of FCD results
            
        Returns:
        --------
        plt.Figure : Matplotlib figure
        """
        fig = plt.figure(figsize=(14, 10))
        
        # Extract states
        A_states = np.array([r['A_t'] for r in results_history])
        A_prime_states = np.array([r['A_prime'] for r in results_history])
        
        # 3D trajectory
        ax1 = fig.add_subplot(221, projection='3d')
        ax1.plot(A_states[:, 0], A_states[:, 1], A_states[:, 2], 
                'b-', alpha=0.6, linewidth=2, label='A_t trajectory')
        ax1.scatter(A_states[0, 0], A_states[0, 1], A_states[0, 2], 
                   c='green', s=100, marker='o', label='Start')
        ax1.scatter(A_states[-1, 0], A_states[-1, 1], A_states[-1, 2], 
                   c='red', s=100, marker='x', label='End')
        ax1.set_xlabel('Trend', fontsize=10)
        ax1.set_ylabel('Momentum', fontsize=10)
        ax1.set_zlabel('Volatility', fontsize=10)
        ax1.set_title('3D State Space Trajectory', fontsize=11, fontweight='bold')
        ax1.legend(fontsize=8)
        
        # Trend-Momentum plane
        ax2 = fig.add_subplot(222)
        scatter = ax2.scatter(A_states[:, 0], A_states[:, 1], 
                             c=np.arange(len(A_states)), cmap='viridis', 
                             s=50, alpha=0.7)
        ax2.plot(A_states[:, 0], A_states[:, 1], 'k-', alpha=0.3, linewidth=1)
        ax2.scatter(A_states[0, 0], A_states[0, 1], c='green', s=150, marker='o', edgecolors='black', linewidths=2)
        ax2.scatter(A_states[-1, 0], A_states[-1, 1], c='red', s=150, marker='x', linewidths=3)
        ax2.set_xlabel('Trend', fontsize=10)
        ax2.set_ylabel('Momentum', fontsize=10)
        ax2.set_title('Trend-Momentum Plane', fontsize=11, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax2, label='Time')
        
        # Momentum-Volatility plane
        ax3 = fig.add_subplot(223)
        scatter2 = ax3.scatter(A_states[:, 1], A_states[:, 2], 
                              c=np.arange(len(A_states)), cmap='plasma', 
                              s=50, alpha=0.7)
        ax3.plot(A_states[:, 1], A_states[:, 2], 'k-', alpha=0.3, linewidth=1)
        ax3.scatter(A_states[0, 1], A_states[0, 2], c='green', s=150, marker='o', edgecolors='black', linewidths=2)
        ax3.scatter(A_states[-1, 1], A_states[-1, 2], c='red', s=150, marker='x', linewidths=3)
        ax3.set_xlabel('Momentum', fontsize=10)
        ax3.set_ylabel('Volatility', fontsize=10)
        ax3.set_title('Momentum-Volatility Plane', fontsize=11, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        plt.colorbar(scatter2, ax=ax3, label='Time')
        
        # Trend-Volatility plane
        ax4 = fig.add_subplot(224)
        scatter3 = ax4.scatter(A_states[:, 0], A_states[:, 2], 
                              c=np.arange(len(A_states)), cmap='coolwarm', 
                              s=50, alpha=0.7)
        ax4.plot(A_states[:, 0], A_states[:, 2], 'k-', alpha=0.3, linewidth=1)
        ax4.scatter(A_states[0, 0], A_states[0, 2], c='green', s=150, marker='o', edgecolors='black', linewidths=2)
        ax4.scatter(A_states[-1, 0], A_states[-1, 2], c='red', s=150, marker='x', linewidths=3)
        ax4.set_xlabel('Trend', fontsize=10)
        ax4.set_ylabel('Volatility', fontsize=10)
        ax4.set_title('Trend-Volatility Plane', fontsize=11, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        plt.colorbar(scatter3, ax=ax4, label='Time')
        
        plt.suptitle('FCD State Space Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def plot_multi_scale_coherence(self, coherence_history: List[Dict]) -> plt.Figure:
        """
        Plot multi-scale coherence metrics.
        
        Parameters:
        -----------
        coherence_history : List[Dict]
            History of coherence metrics
            
        Returns:
        --------
        plt.Figure : Matplotlib figure
        """
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        
        n_points = len(coherence_history)
        time_axis = np.arange(n_points)
        
        # Extract metrics
        tension_coh = np.array([c['tension_coherence'] for c in coherence_history])
        state_coh = np.array([c['state_coherence'] for c in coherence_history])
        trend_coh = np.array([c['trend_coherence'] for c in coherence_history])
        overall_coh = np.array([c['overall_coherence'] for c in coherence_history])
        
        # Tension coherence
        axes[0].plot(time_axis, tension_coh, 'r-', linewidth=2, label='Tension Coherence')
        axes[0].axhline(y=0, color='k', linestyle='--', alpha=0.5)
        axes[0].fill_between(time_axis, 0, tension_coh, where=(tension_coh > 0), alpha=0.3, color='green')
        axes[0].fill_between(time_axis, 0, tension_coh, where=(tension_coh < 0), alpha=0.3, color='red')
        axes[0].set_ylabel('Coherence', fontsize=11)
        axes[0].set_title('Tension Coherence (C_t Alignment)', fontsize=12, fontweight='bold')
        axes[0].legend(fontsize=9)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim([-1.1, 1.1])
        
        # State coherence
        axes[1].plot(time_axis, state_coh, 'b-', linewidth=2, label='State Coherence')
        axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
        axes[1].fill_between(time_axis, 0, state_coh, where=(state_coh > 0), alpha=0.3, color='green')
        axes[1].fill_between(time_axis, 0, state_coh, where=(state_coh < 0), alpha=0.3, color='red')
        axes[1].set_ylabel('Coherence', fontsize=11)
        axes[1].set_title("State Coherence (A'_t Alignment)", fontsize=12, fontweight='bold')
        axes[1].legend(fontsize=9)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim([-1.1, 1.1])
        
        # Overall coherence
        axes[2].plot(time_axis, overall_coh, 'purple', linewidth=2.5, label='Overall Coherence')
        axes[2].plot(time_axis, trend_coh, 'g--', linewidth=1.5, label='Trend Coherence', alpha=0.7)
        axes[2].axhline(y=0, color='k', linestyle='--', alpha=0.5)
        axes[2].fill_between(time_axis, 0, overall_coh, alpha=0.3, color='purple')
        axes[2].set_xlabel('Time', fontsize=11)
        axes[2].set_ylabel('Coherence', fontsize=11)
        axes[2].set_title('Overall Multi-Scale Coherence', fontsize=12, fontweight='bold')
        axes[2].legend(fontsize=9)
        axes[2].grid(True, alpha=0.3)
        
        plt.suptitle('Multi-Scale FCD Coherence Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def plot_backtest_equity_curve(self,
                                   records: 'pd.DataFrame',
                                   ticker: str = "",
                                   mode: str = "") -> plt.Figure:
        """
        Plot equity curve from backtest results.
        
        Parameters:
        -----------
        records : pd.DataFrame
            Backtest records with equity column
        ticker : str
            Ticker symbol for title
        mode : str
            Strategy mode for title
            
        Returns:
        --------
        plt.Figure : Matplotlib figure
        """
        import pandas as pd
        
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        
        dates = records['date'].values
        equity = records['equity'].values
        
        # Calculate drawdown
        running_max = np.maximum.accumulate(equity)
        drawdown = (equity - running_max) / running_max
        
        # 1. Equity curve
        axes[0].plot(dates, equity, 'b-', linewidth=2, label='Equity')
        axes[0].fill_between(dates, 1.0, equity, where=(equity >= 1.0), alpha=0.3, color='green')
        axes[0].fill_between(dates, 1.0, equity, where=(equity < 1.0), alpha=0.3, color='red')
        axes[0].axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='Initial Capital')
        axes[0].set_ylabel('Equity', fontsize=11)
        axes[0].set_title(f'Equity Curve - {ticker} - Mode {mode}', fontsize=12, fontweight='bold')
        axes[0].legend(fontsize=9)
        axes[0].grid(True, alpha=0.3)
        
        # 2. Drawdown
        axes[1].fill_between(dates, 0, drawdown, alpha=0.5, color='red')
        axes[1].plot(dates, drawdown, 'r-', linewidth=1.5, label='Drawdown')
        axes[1].set_ylabel('Drawdown', fontsize=11)
        axes[1].set_title('Drawdown', fontsize=12, fontweight='bold')
        axes[1].legend(fontsize=9)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim([min(drawdown.min() * 1.1, -0.01), 0.01])
        
        # 3. Cumulative P&L
        cumulative_pnl = records['pnl'].cumsum().values
        axes[2].plot(dates, cumulative_pnl, 'g-', linewidth=2, label='Cumulative P&L')
        axes[2].fill_between(dates, 0, cumulative_pnl, where=(cumulative_pnl >= 0), alpha=0.3, color='green')
        axes[2].fill_between(dates, 0, cumulative_pnl, where=(cumulative_pnl < 0), alpha=0.3, color='red')
        axes[2].axhline(y=0, color='k', linestyle='--', alpha=0.5)
        axes[2].set_xlabel('Date', fontsize=11)
        axes[2].set_ylabel('Cumulative P&L', fontsize=11)
        axes[2].set_title('Cumulative P&L', fontsize=12, fontweight='bold')
        axes[2].legend(fontsize=9)
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_backtest_regime_performance(self,
                                        records: 'pd.DataFrame') -> plt.Figure:
        """
        Plot performance by regime (volatility and tension).
        
        Parameters:
        -----------
        records : pd.DataFrame
            Backtest records
            
        Returns:
        --------
        plt.Figure : Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Split by volatility
        median_vol = records['A_t_volatility'].median()
        high_vol = records[records['A_t_volatility'] >= median_vol]
        low_vol = records[records['A_t_volatility'] < median_vol]
        
        # Split by tension
        median_tension = records['C_magnitude'].median()
        high_tension = records[records['C_magnitude'] >= median_tension]
        low_tension = records[records['C_magnitude'] < median_tension]
        
        # Helper function to plot regime stats
        def plot_regime_bars(ax, data_dict, title):
            regimes = list(data_dict.keys())
            accuracies = [data_dict[r]['accuracy'] for r in regimes]
            mean_pnls = [data_dict[r]['mean_pnl'] * 100 for r in regimes]
            n_trades = [data_dict[r]['n_trades'] for r in regimes]
            
            x = np.arange(len(regimes))
            width = 0.35
            
            ax2 = ax.twinx()
            bars1 = ax.bar(x - width/2, accuracies, width, label='Accuracy', color='blue', alpha=0.7)
            bars2 = ax2.bar(x + width/2, mean_pnls, width, label='Mean PnL (%)', color='green', alpha=0.7)
            
            ax.set_ylabel('Accuracy', fontsize=10)
            ax2.set_ylabel('Mean PnL (%)', fontsize=10)
            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(regimes, rotation=45, ha='right')
            ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
            ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax.legend(loc='upper left', fontsize=8)
            ax2.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add trade count labels
            for i, (bar, n) in enumerate(zip(bars1, n_trades)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'n={n}', ha='center', va='bottom', fontsize=7)
        
        # Calculate metrics for each regime
        def get_regime_metrics(subset):
            predictions = subset[subset['predicted_direction'] != 0]
            accuracy = predictions['correct'].fillna(False).mean() if len(predictions) > 0 else 0.0
            mean_pnl = subset['pnl'].fillna(0).mean()
            n_trades = (subset['position'] != 0).sum()
            return {'accuracy': accuracy, 'mean_pnl': mean_pnl, 'n_trades': n_trades}
        
        # Volatility regimes
        vol_data = {
            'High Vol': get_regime_metrics(high_vol),
            'Low Vol': get_regime_metrics(low_vol)
        }
        plot_regime_bars(axes[0, 0], vol_data, 'Performance by Volatility Regime')
        
        # Tension regimes
        tension_data = {
            'High Tension': get_regime_metrics(high_tension),
            'Low Tension': get_regime_metrics(low_tension)
        }
        plot_regime_bars(axes[0, 1], tension_data, 'Performance by Tension Regime')
        
        # Scatter: Volatility vs PnL
        axes[1, 0].scatter(records['A_t_volatility'], records['pnl'] * 100, 
                          alpha=0.3, c=records['position'], cmap='RdYlGn', s=20)
        axes[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.5)
        axes[1, 0].axvline(x=median_vol, color='r', linestyle='--', alpha=0.5, label='Median Vol')
        axes[1, 0].set_xlabel('Volatility', fontsize=10)
        axes[1, 0].set_ylabel('PnL (%)', fontsize=10)
        axes[1, 0].set_title('Volatility vs P&L', fontsize=11, fontweight='bold')
        axes[1, 0].legend(fontsize=8)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Scatter: Tension vs PnL
        axes[1, 1].scatter(records['C_magnitude'], records['pnl'] * 100,
                          alpha=0.3, c=records['position'], cmap='RdYlGn', s=20)
        axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
        axes[1, 1].axvline(x=median_tension, color='r', linestyle='--', alpha=0.5, label='Median Tension')
        axes[1, 1].set_xlabel('Tension (C magnitude)', fontsize=10)
        axes[1, 1].set_ylabel('PnL (%)', fontsize=10)
        axes[1, 1].set_title('Tension vs P&L', fontsize=11, fontweight='bold')
        axes[1, 1].legend(fontsize=8)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_backtest_calibration(self,
                                  records: 'pd.DataFrame') -> plt.Figure:
        """
        Plot probability calibration curve.
        
        Shows whether predicted probabilities match empirical frequencies.
        
        Parameters:
        -----------
        records : pd.DataFrame
            Backtest records with upward_prob and realized_direction
            
        Returns:
        --------
        plt.Figure : Matplotlib figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Define probability bins
        bins = [0.0, 0.5, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0]
        
        expected_probs = []
        empirical_freqs = []
        bin_counts = []
        
        for i in range(len(bins) - 1):
            bin_low = bins[i]
            bin_high = bins[i + 1]
            
            in_bin = records[
                (records['upward_prob'] >= bin_low) & 
                (records['upward_prob'] < bin_high)
            ]
            
            if len(in_bin) == 0:
                continue
            
            actual_up = (in_bin['realized_direction'] == 1).sum()
            empirical_freq = actual_up / len(in_bin)
            expected_prob = (bin_low + bin_high) / 2
            
            expected_probs.append(expected_prob)
            empirical_freqs.append(empirical_freq)
            bin_counts.append(len(in_bin))
        
        # 1. Calibration curve
        if len(expected_probs) > 0:
            axes[0].plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=2, alpha=0.5)
            axes[0].scatter(expected_probs, empirical_freqs, s=100, alpha=0.7, 
                          c=bin_counts, cmap='viridis', edgecolors='black', linewidth=1)
            axes[0].plot(expected_probs, empirical_freqs, 'b-', alpha=0.5, linewidth=1)
            
            # Add colorbar
            cbar = plt.colorbar(axes[0].collections[0], ax=axes[0])
            cbar.set_label('Sample Count', fontsize=9)
            
            axes[0].set_xlabel('Predicted Probability (Upward)', fontsize=11)
            axes[0].set_ylabel('Empirical Frequency (Upward)', fontsize=11)
            axes[0].set_title('Probability Calibration Curve', fontsize=12, fontweight='bold')
            axes[0].legend(fontsize=9)
            axes[0].grid(True, alpha=0.3)
            axes[0].set_xlim([0, 1])
            axes[0].set_ylim([0, 1])
        else:
            axes[0].text(0.5, 0.5, 'Insufficient data for calibration',
                        ha='center', va='center', fontsize=12)
        
        # 2. Bin counts histogram
        if len(expected_probs) > 0:
            axes[1].bar(range(len(bin_counts)), bin_counts, color='steelblue', alpha=0.7)
            axes[1].set_xlabel('Probability Bin', fontsize=11)
            axes[1].set_ylabel('Sample Count', fontsize=11)
            axes[1].set_title('Sample Distribution Across Bins', fontsize=12, fontweight='bold')
            axes[1].set_xticks(range(len(expected_probs)))
            axes[1].set_xticklabels([f'{p:.2f}' for p in expected_probs], rotation=45)
            axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        return fig
