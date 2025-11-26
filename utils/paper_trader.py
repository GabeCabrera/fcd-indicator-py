"""
Paper Trading Engine
====================
Simulates buy/sell logic with CSV logging and P/L tracking.

Features:
- Single position long/flat model
- Entry price tracking
- P/L calculation
- CSV trade logging with metadata
- Real-time position management
"""

import sys
import csv
from datetime import datetime
from typing import Dict, Optional
from pathlib import Path


class PaperTrader:
    """
    Paper trading engine for FCD signals.
    
    Manages:
    - Cash and equity tracking
    - Position state (long/flat)
    - Trade execution and logging
    - P/L calculation
    """
    
    def __init__(
        self,
        initial_cash: float = 100000.0,
        position_size: float = 0.95,
        csv_file: str = "trades.csv"
    ):
        """
        Initialize paper trader.
        
        Args:
            initial_cash: Starting cash balance
            position_size: Fraction of cash to use per trade (0.0 to 1.0)
            csv_file: Path to CSV file for trade logging
        """
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.position_size = position_size
        self.csv_file = csv_file
        
        # Position tracking: {symbol: {'shares': N, 'entry_price': P, 'entry_time': T}}
        self.positions: Dict[str, Dict] = {}
        
        # Trade history
        self.trades: list = []
        
        # Initialize CSV file
        self._init_csv()
        
        print(f"[PaperTrader] Initialized:")
        print(f"  Initial Cash: ${self.cash:,.2f}")
        print(f"  Position Size: {self.position_size*100:.0f}%")
        print(f"  CSV Log: {self.csv_file}")
    
    def _init_csv(self):
        """Initialize CSV file with headers"""
        csv_path = Path(self.csv_file)
        
        # Create file with headers if it doesn't exist
        if not csv_path.exists():
            with open(self.csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp',
                    'action',
                    'symbol',
                    'price',
                    'shares',
                    'entry_price',
                    'pnl',
                    'pnl_pct',
                    'cash',
                    'equity',
                    'fcd_value',
                    'becoming_score',
                    'confidence',
                    'reason'
                ])
            print(f"[PaperTrader] Created CSV log: {self.csv_file}")
    
    def _log_trade(self, trade_data: Dict):
        """Append trade to CSV log"""
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.fromtimestamp(trade_data['timestamp']/1000).strftime('%Y-%m-%d %H:%M:%S'),
                trade_data['action'],
                trade_data['symbol'],
                f"{trade_data['price']:.2f}",
                trade_data.get('shares', ''),
                f"{trade_data.get('entry_price', 0):.2f}",
                f"{trade_data.get('pnl', 0):.2f}",
                f"{trade_data.get('pnl_pct', 0):.2f}",
                f"{trade_data['cash']:.2f}",
                f"{trade_data['equity']:.2f}",
                f"{trade_data.get('fcd_value', ''):.4f}" if trade_data.get('fcd_value') else '',
                f"{trade_data.get('becoming_score', ''):.4f}" if trade_data.get('becoming_score') else '',
                trade_data.get('confidence', ''),
                trade_data.get('reason', '')
            ])
        sys.stdout.flush()
    
    def get_position(self, symbol: str) -> Optional[Dict]:
        """Get current position for symbol"""
        return self.positions.get(symbol)
    
    def get_equity(self, current_prices: Optional[Dict[str, float]] = None) -> float:
        """
        Calculate current equity (cash + position values).
        
        Args:
            current_prices: Dict of {symbol: current_price} for open positions
        
        Returns:
            Total equity
        """
        equity = self.cash
        
        if current_prices:
            for symbol, position in self.positions.items():
                current_price = current_prices.get(symbol)
                if current_price:
                    position_value = position['shares'] * current_price
                    equity += position_value
        
        return equity
    
    def buy(
        self,
        symbol: str,
        price: float,
        timestamp: int,
        metadata: Optional[Dict] = None
    ) -> Dict:
        """
        Execute BUY order (open long position).
        
        Args:
            symbol: Trading symbol
            price: Execution price
            timestamp: Unix timestamp in milliseconds
            metadata: FCD signal metadata
        
        Returns:
            Trade result dictionary
        """
        # Check if already in position
        if symbol in self.positions:
            print(f"[PaperTrader] Already LONG {symbol}, ignoring BUY signal")
            return None
        
        # Calculate position size
        trade_value = self.cash * self.position_size
        shares = trade_value / price
        cost = shares * price
        
        # Check sufficient funds
        if cost > self.cash:
            print(f"[PaperTrader] Insufficient cash for {symbol}: Need ${cost:.2f}, Have ${self.cash:.2f}")
            return None
        
        # Execute trade
        self.cash -= cost
        self.positions[symbol] = {
            'shares': shares,
            'entry_price': price,
            'entry_time': timestamp
        }
        
        # Calculate equity
        equity = self.get_equity({symbol: price})
        
        # Create trade record
        trade_data = {
            'timestamp': timestamp,
            'action': 'BUY',
            'symbol': symbol,
            'price': price,
            'shares': shares,
            'entry_price': price,
            'cash': self.cash,
            'equity': equity,
            'fcd_value': metadata.get('fcd_value') if metadata else None,
            'becoming_score': metadata.get('becoming_score') if metadata else None,
            'confidence': metadata.get('confidence') if metadata else None,
            'reason': metadata.get('reason') if metadata else None
        }
        
        # Log trade
        self._log_trade(trade_data)
        self.trades.append(trade_data)
        
        print(f"[PaperTrader] BUY {shares:.4f} shares of {symbol} @ ${price:.2f}")
        print(f"[PaperTrader] Cash: ${self.cash:.2f} | Equity: ${equity:.2f}")
        sys.stdout.flush()
        
        return trade_data
    
    def sell(
        self,
        symbol: str,
        price: float,
        timestamp: int,
        metadata: Optional[Dict] = None
    ) -> Dict:
        """
        Execute SELL order (close long position).
        
        Args:
            symbol: Trading symbol
            price: Execution price
            timestamp: Unix timestamp in milliseconds
            metadata: FCD signal metadata
        
        Returns:
            Trade result dictionary
        """
        # Check if in position
        if symbol not in self.positions:
            print(f"[PaperTrader] No position in {symbol}, ignoring SELL signal")
            return None
        
        # Get position details
        position = self.positions[symbol]
        shares = position['shares']
        entry_price = position['entry_price']
        
        # Calculate proceeds and P/L
        proceeds = shares * price
        cost_basis = shares * entry_price
        pnl = proceeds - cost_basis
        pnl_pct = (pnl / cost_basis) * 100
        
        # Execute trade
        self.cash += proceeds
        del self.positions[symbol]
        
        # Calculate equity
        equity = self.get_equity()
        
        # Create trade record
        trade_data = {
            'timestamp': timestamp,
            'action': 'SELL',
            'symbol': symbol,
            'price': price,
            'shares': shares,
            'entry_price': entry_price,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'cash': self.cash,
            'equity': equity,
            'fcd_value': metadata.get('fcd_value') if metadata else None,
            'becoming_score': metadata.get('becoming_score') if metadata else None,
            'confidence': metadata.get('confidence') if metadata else None,
            'reason': metadata.get('reason') if metadata else None
        }
        
        # Log trade
        self._log_trade(trade_data)
        self.trades.append(trade_data)
        
        print(f"[PaperTrader] SELL {shares:.4f} shares of {symbol} @ ${price:.2f}")
        print(f"[PaperTrader] P&L: ${pnl:.2f} ({pnl_pct:+.2f}%)")
        print(f"[PaperTrader] Cash: ${self.cash:.2f} | Equity: ${equity:.2f}")
        sys.stdout.flush()
        
        return trade_data
    
    def get_stats(self) -> Dict:
        """Get trading statistics"""
        total_trades = len(self.trades)
        buy_trades = [t for t in self.trades if t['action'] == 'BUY']
        sell_trades = [t for t in self.trades if t['action'] == 'SELL']
        
        # Calculate P/L stats
        total_pnl = sum(t.get('pnl', 0) for t in sell_trades)
        winning_trades = [t for t in sell_trades if t.get('pnl', 0) > 0]
        losing_trades = [t for t in sell_trades if t.get('pnl', 0) < 0]
        
        win_rate = len(winning_trades) / len(sell_trades) if sell_trades else 0
        avg_win = sum(t['pnl'] for t in winning_trades) / len(winning_trades) if winning_trades else 0
        avg_loss = sum(t['pnl'] for t in losing_trades) / len(losing_trades) if losing_trades else 0
        
        # Current equity
        equity = self.get_equity()
        total_return = equity - self.initial_cash
        total_return_pct = (total_return / self.initial_cash) * 100
        
        return {
            'initial_cash': self.initial_cash,
            'current_cash': self.cash,
            'current_equity': equity,
            'total_return': total_return,
            'total_return_pct': total_return_pct,
            'total_trades': total_trades,
            'completed_trades': len(sell_trades),
            'open_positions': len(self.positions),
            'total_pnl': total_pnl,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'positions': {
                symbol: {
                    'shares': pos['shares'],
                    'entry_price': pos['entry_price'],
                    'entry_time': datetime.fromtimestamp(pos['entry_time']/1000).isoformat()
                }
                for symbol, pos in self.positions.items()
            }
        }
    
    def reset(self):
        """Reset trader to initial state"""
        self.cash = self.initial_cash
        self.positions = {}
        self.trades = []
        
        print(f"[PaperTrader] Reset to initial state")
        print(f"[PaperTrader] Cash: ${self.cash:,.2f}")
        sys.stdout.flush()


if __name__ == "__main__":
    # Test paper trader
    trader = PaperTrader(initial_cash=100000, position_size=0.95)
    
    # Simulate trades
    print("\n=== Testing Paper Trader ===\n")
    
    # Buy MGC
    metadata = {
        'fcd_value': 0.65,
        'becoming_score': 0.184,
        'confidence': 0.75,
        'reason': 'fcd_above_threshold'
    }
    
    timestamp = int(datetime.now().timestamp() * 1000)
    
    trader.buy('MGC=F', 2450.50, timestamp, metadata)
    
    # Simulate price movement
    import time
    time.sleep(1)
    
    # Sell MGC
    timestamp = int(datetime.now().timestamp() * 1000)
    trader.sell('MGC=F', 2455.75, timestamp, metadata)
    
    # Show stats
    print("\n=== Trading Stats ===")
    stats = trader.get_stats()
    for key, value in stats.items():
        if key != 'positions':
            print(f"{key}: {value}")
    
    print("\n✅ Paper trader test complete")
