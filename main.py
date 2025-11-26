"""
FCD Cloud Trading Engine - Main Server
======================================
FastAPI webhook server that receives TradingView bar data and generates
FCD trading signals using the complete FCD-PSE model.

Deployment: Railway
Entry: uvicorn main:app --host 0.0.0.0 --port 8000
"""

import sys
import os
from datetime import datetime
from typing import Dict, Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np

# Import FCD components
from fcd.signal.fcd_signal_generator import FCDSignalGenerator
from utils.paper_trader import PaperTrader

# Initialize FastAPI
app = FastAPI(title="FCD Trading Engine", version="1.0.0")

# Global state
fcd_engine: Optional[FCDSignalGenerator] = None
paper_trader: Optional[PaperTrader] = None
bar_cache: Dict[str, list] = {}


class TradingViewBar(BaseModel):
    """TradingView webhook payload schema"""
    symbol: str = Field(..., description="Trading symbol (e.g., SPY, MGC=F)")
    timestamp: int = Field(..., description="Unix timestamp in milliseconds")
    open: float = Field(..., gt=0)
    high: float = Field(..., gt=0)
    low: float = Field(..., gt=0)
    close: float = Field(..., gt=0)
    volume: float = Field(..., ge=0)


@app.on_event("startup")
async def startup_event():
    """Initialize FCD engine and paper trader on server startup"""
    global fcd_engine, paper_trader
    
    print("\n" + "="*60)
    print("FCD CLOUD TRADING ENGINE - STARTUP")
    print("="*60)
    
    # Initialize FCD Signal Generator
    print("\n[1/3] Initializing FCD Signal Generator...")
    try:
        fcd_engine = FCDSignalGenerator(
            min_becoming_score=0.0,  # Use ranking CSV to filter
            lookback_bars=100,
            fcd_long_threshold=0.1,  # Configurable threshold
            fcd_short_threshold=0.1,
            allow_shorts=False  # Long-only for now
        )
        
        # Load BecomingScore rankings
        rankings_dir = Path("fcd/rankings")
        if rankings_dir.exists():
            fcd_engine.load_becoming_scores(str(rankings_dir))
        else:
            print(f"⚠️  Rankings directory not found: {rankings_dir}")
            print("    Server will accept all symbols without BecomingScore filtering")
        
        print("✅ FCD Signal Generator initialized")
        
    except Exception as e:
        print(f"❌ Failed to initialize FCD Signal Generator: {e}")
        raise
    
    # Initialize Paper Trader
    print("\n[2/3] Initializing Paper Trading Engine...")
    try:
        paper_trader = PaperTrader(
            initial_cash=100000.0,
            position_size=0.95,  # 95% of cash per trade
            csv_file="trades.csv"
        )
        print("✅ Paper Trading Engine initialized")
        print(f"    Initial Cash: ${paper_trader.cash:,.2f}")
        
    except Exception as e:
        print(f"❌ Failed to initialize Paper Trader: {e}")
        raise
    
    # Show tradeable instruments
    print("\n[3/3] Tradeable Instruments:")
    tradeable = fcd_engine.get_tradeable_instruments()
    if tradeable:
        for sym in tradeable[:10]:  # Show top 10
            score = fcd_engine.becoming_scores.get(sym, 0)
            print(f"    {sym}: BecomingScore = {score:.3f}")
        if len(tradeable) > 10:
            print(f"    ... and {len(tradeable) - 10} more")
    else:
        print("    ⚠️  No ranking data loaded - accepting all symbols")
    
    print("\n" + "="*60)
    print("SERVER READY - Listening for TradingView webhooks")
    print("="*60 + "\n")
    sys.stdout.flush()


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "service": "FCD Cloud Trading Engine",
        "version": "1.0.1",
        "commit": "283d01f",
        "last_updated": "2025-11-25",
        "endpoints": {
            "webhook": "/webhook",
            "health": "/health",
            "stats": "/stats"
        }
    }


@app.get("/health")
async def health():
    """Detailed health check"""
    global fcd_engine, paper_trader
    
    return {
        "status": "healthy",
        "fcd_engine": "initialized" if fcd_engine else "not_initialized",
        "paper_trader": "initialized" if paper_trader else "not_initialized",
        "cache_symbols": list(bar_cache.keys()) if bar_cache else [],
        "timestamp": datetime.now().isoformat()
    }


@app.get("/stats")
async def stats():
    """Get trading statistics"""
    global paper_trader
    
    if not paper_trader:
        raise HTTPException(status_code=503, detail="Paper trader not initialized")
    
    return paper_trader.get_stats()


@app.post("/webhook")
async def webhook(bar: TradingViewBar):
    """
    Main webhook endpoint for TradingView alerts
    
    Receives bar data, processes through FCD engine, and executes paper trades
    """
    global fcd_engine, paper_trader, bar_cache
    
    if not fcd_engine or not paper_trader:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    # Log incoming webhook
    print("\n" + "─"*60)
    print(f"📊 INCOMING WEBHOOK - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("─"*60)
    print(f"Symbol: {bar.symbol}")
    print(f"Timestamp: {datetime.fromtimestamp(bar.timestamp/1000).strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"OHLCV: O={bar.open:.2f} H={bar.high:.2f} L={bar.low:.2f} C={bar.close:.2f} V={bar.volume:,.0f}")
    sys.stdout.flush()
    
    # Convert bar to dictionary
    bar_data = {
        'timestamp': bar.timestamp,
        'open': bar.open,
        'high': bar.high,
        'low': bar.low,
        'close': bar.close,
        'volume': bar.volume
    }
    
    # Update bar cache
    if bar.symbol not in bar_cache:
        bar_cache[bar.symbol] = []
    bar_cache[bar.symbol].append(bar_data)
    
    # Keep only last 200 bars
    if len(bar_cache[bar.symbol]) > 200:
        bar_cache[bar.symbol] = bar_cache[bar.symbol][-200:]
    
    print(f"\n📈 Bar Cache: {len(bar_cache[bar.symbol])} bars for {bar.symbol}")
    sys.stdout.flush()
    
    # Get current position
    current_position = paper_trader.get_position(bar.symbol)
    position_state = "LONG" if current_position else "FLAT"
    
    # Generate FCD signal
    print(f"\n🔬 FCD ANALYSIS")
    print(f"Position: {position_state}")
    sys.stdout.flush()
    
    try:
        signal, metadata = fcd_engine.get_signal(
            symbol=bar.symbol,
            bar_data=bar_data,
            current_position=position_state
        )
        
        # Log FCD results
        print(f"\n📊 FCD OUTPUT:")
        if metadata:
            print(f"  Signal: {signal}")
            print(f"  FCD Value: {metadata.get('fcd_value', 'N/A'):.4f}")
            print(f"  BecomingScore: {metadata.get('becoming_score', 'N/A'):.4f}")
            print(f"  Confidence: {metadata.get('confidence', 'N/A')}")
            print(f"  Reason: {metadata.get('reason', 'N/A')}")
        else:
            print(f"  Signal: {signal} (no metadata)")
        sys.stdout.flush()
        
        # Execute trade based on signal
        trade_result = None
        if signal == "LONG" and not current_position:
            # Open long position
            print(f"\n🟢 EXECUTING BUY ORDER")
            sys.stdout.flush()
            trade_result = paper_trader.buy(
                symbol=bar.symbol,
                price=bar.close,
                timestamp=bar.timestamp,
                metadata=metadata
            )
            
        elif signal == "FLAT" and current_position:
            # Close long position
            print(f"\n🔴 EXECUTING SELL ORDER")
            sys.stdout.flush()
            trade_result = paper_trader.sell(
                symbol=bar.symbol,
                price=bar.close,
                timestamp=bar.timestamp,
                metadata=metadata
            )
        
        elif signal == "HOLD":
            print(f"\n⏸️  HOLD - No action taken")
            sys.stdout.flush()
        
        # Log trade result
        if trade_result:
            print(f"\n💰 TRADE EXECUTED:")
            print(f"  Action: {trade_result['action']}")
            print(f"  Price: ${trade_result['price']:.2f}")
            print(f"  Shares: {trade_result.get('shares', 'N/A')}")
            if 'pnl' in trade_result:
                pnl_pct = (trade_result['pnl'] / trade_result['entry_price']) * 100
                print(f"  P&L: ${trade_result['pnl']:.2f} ({pnl_pct:+.2f}%)")
            print(f"  Cash: ${trade_result['cash']:.2f}")
            print(f"  Equity: ${trade_result['equity']:.2f}")
            sys.stdout.flush()
        
        print("\n" + "─"*60 + "\n")
        sys.stdout.flush()
        
        # Return response
        return {
            "status": "success",
            "symbol": bar.symbol,
            "signal": signal,
            "metadata": metadata,
            "trade": trade_result,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


@app.get("/cache/{symbol}")
async def get_cache(symbol: str):
    """Get cached bar data for a symbol"""
    global bar_cache
    
    if symbol not in bar_cache:
        raise HTTPException(status_code=404, detail=f"No cache data for {symbol}")
    
    return {
        "symbol": symbol,
        "bars": len(bar_cache[symbol]),
        "data": bar_cache[symbol]
    }


@app.post("/reset")
async def reset_trader():
    """Reset paper trader (admin endpoint)"""
    global paper_trader
    
    if not paper_trader:
        raise HTTPException(status_code=503, detail="Paper trader not initialized")
    
    paper_trader.reset()
    
    return {
        "status": "success",
        "message": "Paper trader reset",
        "timestamp": datetime.now().isoformat()
    }


if __name__ == "__main__":
    import uvicorn
    
    # Run server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
