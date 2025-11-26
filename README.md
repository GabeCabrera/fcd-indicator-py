# FCD Cloud Trading Engine

**Complete cloud-ready FCD-PSE trading system for Railway deployment**

## 🎯 Overview

This is a production-ready FastAPI server that:

✅ Receives TradingView webhook alerts with bar data (OHLCV)  
✅ Processes bars through the complete FCD-PSE quantum causality model  
✅ Generates real trading signals (LONG/FLAT) with full BecomingScore filtering  
✅ Executes paper trades with P/L tracking  
✅ Logs everything to CSV and Railway console  
✅ Runs 24/7 on Railway (no local computer needed)

## 📁 Project Structure

```
fcd-cloud-server/
│
├── main.py                          # FastAPI webhook server
├── requirements.txt                 # Python dependencies
├── Procfile                         # Railway startup command
├── railway.json                     # Railway configuration
├── runtime.txt                      # Python version
├── trades.csv                       # Auto-generated trade log
│
├── fcd/
│   ├── core/                        # Complete FCD-PSE model (DO NOT MODIFY)
│   │   ├── fcd_indicator.py        # Main FCD indicator
│   │   ├── fcd_state.py            # FCD state transformation
│   │   ├── probabilistic.py        # Monte Carlo prediction
│   │   ├── kalman.py               # Kalman filtering
│   │   ├── monte_carlo.py          # Path generation
│   │   ├── primitives.py           # Math primitives
│   │   ├── multi_scale.py          # Multi-timeframe
│   │   └── btc_mode_config.py      # Regime configuration
│   │
│   ├── signal/
│   │   └── fcd_signal_generator.py # Signal generation logic
│   │
│   └── rankings/
│       └── consolidated_futures.csv # BecomingScore rankings
│
└── utils/
    └── paper_trader.py              # Paper trading engine
```

## 🚀 Railway Deployment Guide

### Step 1: Prepare Your Repository

1. **Navigate to the cloud server directory:**
   ```bash
   cd fcd-cloud-server
   ```

2. **Initialize git (if not already):**
   ```bash
   git init
   git add .
   git commit -m "Initial FCD cloud server"
   ```

3. **Push to GitHub:**
   ```bash
   # Create new repo on GitHub first, then:
   git remote add origin https://github.com/YOUR_USERNAME/fcd-cloud-server.git
   git push -u origin main
   ```

### Step 2: Deploy to Railway

1. **Go to [Railway.app](https://railway.app)**

2. **Create New Project:**
   - Click "New Project"
   - Select "Deploy from GitHub repo"
   - Authorize Railway to access your repositories
   - Select `fcd-cloud-server` repository

3. **Railway will automatically:**
   - Detect Python project
   - Install dependencies from `requirements.txt`
   - Run the command from `Procfile`: `uvicorn main:app --host 0.0.0.0 --port $PORT`

4. **Get Your Webhook URL:**
   - Once deployed, Railway provides a public URL
   - Example: `https://fcd-cloud-server-production.up.railway.app`
   - Your webhook endpoint: `https://YOUR-URL/webhook`

5. **Monitor Logs:**
   - Click "View Logs" in Railway dashboard
   - All FCD signals and trades appear here in real-time

### Step 3: Environment Variables (Optional)

You can configure these in Railway's Variables section:

```
FCD_LONG_THRESHOLD=0.1
FCD_SHORT_THRESHOLD=0.1
MIN_BECOMING_SCORE=0.0
INITIAL_CASH=100000
POSITION_SIZE=0.95
```

## 📡 TradingView Alert Setup

### 1. Create Alert in TradingView

1. Open your chart (e.g., SPY, MGC, BTC)
2. Click the Alert button (🔔)
3. Set your conditions (e.g., "Bar Close")
4. Configure webhook:

### 2. Webhook URL
```
https://YOUR-RAILWAY-URL/webhook
```

### 3. Webhook Message (JSON)

Use this exact format:

```json
{
  "symbol": "{{ticker}}",
  "timestamp": {{timenow}},
  "open": {{open}},
  "high": {{high}},
  "low": {{low}},
  "close": {{close}},
  "volume": {{volume}}
}
```

### 4. Alert Settings

- **Condition:** Once Per Bar Close
- **Frequency:** Every Time
- **Expiration:** Open-ended

### 5. Test Alert

Send a test alert and check Railway logs for:
```
📊 INCOMING WEBHOOK
Symbol: SPY
OHLCV: O=450.23 H=451.45 L=449.87 C=450.98 V=65432100
```

## 🔬 Understanding the FCD Engine

### Signal Generation Flow

```
TradingView Alert
    ↓
[1] Webhook receives OHLCV bar
    ↓
[2] Update bar cache (stores last 100-200 bars)
    ↓
[3] Check BecomingScore (filters low-quality instruments)
    ↓
[4] Calculate FCD indicator:
    • FCD State Transformation (A_t → B_t → C_t → X_t → A'_t)
    • Nonlinear Memory
    • Causal Mass
    • Asymmetry Function
    • Regime Classification
    ↓
[5] Generate Trading Signal:
    • LONG: FCD alignment + passing thresholds
    • FLAT: Exit condition or low FCD
    • HOLD: No change in position
    ↓
[6] Execute Paper Trade:
    • BUY: Open long position
    • SELL: Close long position
    ↓
[7] Log to CSV and Railway Console
```

### FCD Metrics Explained

- **FCD Value:** Normalized coherence (0.0 to 1.0)
  - Higher = stronger directional alignment
  - Threshold: 0.1 for LONG signals

- **BecomingScore:** Historical performance metric (-1.0 to 1.0)
  - Filters instruments by past FCD effectiveness
  - Top performers: MGC=F (0.184), SIL=F (0.120)

- **Confidence:** Directional probability spread
  - p_up - p_down
  - Higher = more confident prediction

- **Regime:** Market state classification
  - uptrend_low_vol, uptrend_high_vol
  - downtrend_low_vol, downtrend_high_vol
  - sideways_low_vol, sideways_high_vol

## 📊 Monitoring & Logs

### Railway Console Logs

You'll see detailed output like this:

```
📊 INCOMING WEBHOOK - 2025-11-25 14:30:00
─────────────────────────────────────────
Symbol: MGC=F
Timestamp: 2025-11-25 14:30:00
OHLCV: O=2450.25 H=2455.80 L=2448.90 C=2454.50 V=12450

📈 Bar Cache: 87 bars for MGC=F

🔬 FCD ANALYSIS
Position: FLAT

📊 FCD OUTPUT:
  Signal: LONG
  FCD Value: 0.6543
  BecomingScore: 0.1840
  Confidence: 0.7234
  Reason: fcd_long_signal

🟢 EXECUTING BUY ORDER

💰 TRADE EXECUTED:
  Action: BUY
  Price: $2454.50
  Shares: 38.6789
  Cash: $5,000.00
  Equity: $100,000.00
─────────────────────────────────────────
```

### CSV Trade Log Format

`trades.csv`:

```csv
timestamp,action,symbol,price,shares,entry_price,pnl,pnl_pct,cash,equity,fcd_value,becoming_score,confidence,reason
2025-11-25 14:30:00,BUY,MGC=F,2454.50,38.6789,2454.50,0.00,0.00,5000.00,100000.00,0.6543,0.1840,0.7234,fcd_long_signal
2025-11-25 16:00:00,SELL,MGC=F,2458.75,38.6789,2454.50,164.35,1.73,100164.35,100164.35,0.3210,0.1840,-0.1234,fcd_neutral
```

### Health Check Endpoints

- **`GET /`** - Service info
- **`GET /health`** - Detailed health status
- **`GET /stats`** - Trading statistics (P/L, win rate, etc.)
- **`GET /cache/{symbol}`** - View cached bars for symbol
- **`POST /reset`** - Reset paper trader (admin)

## 🔧 Configuration

### Adjusting FCD Thresholds

Edit `main.py` startup configuration:

```python
fcd_engine = FCDSignalGenerator(
    min_becoming_score=0.0,      # 0.0 = accept all symbols
    lookback_bars=100,            # Bars for FCD calculation
    fcd_long_threshold=0.1,       # Lower = more signals
    fcd_short_threshold=0.1,
    allow_shorts=False,           # Long-only mode
    interval="1d"                 # Daily bars
)
```

### Adjusting Paper Trading

```python
paper_trader = PaperTrader(
    initial_cash=100000.0,        # Starting balance
    position_size=0.95,           # 95% of cash per trade
    csv_file="trades.csv"
)
```

## 📈 BecomingScore Rankings

The engine uses pre-calculated BecomingScore rankings from:
`fcd/rankings/consolidated_futures.csv`

### Current Top Performers:

| Ticker | BecomingScore | Status |
|--------|---------------|--------|
| MGC=F  | 0.184        | ✓ Top  |
| SIL=F  | 0.120        | ✓ Good |
| MNQ=F  | 0.014        | ○ Mid  |
| MES=F  | 0.007        | ○ Mid  |

### Update Rankings:

1. Run your local FCD ranking script
2. Copy new `consolidated_futures.csv` to `fcd/rankings/`
3. Redeploy to Railway

## 🐛 Troubleshooting

### Issue: No signals generated

**Check:**
1. Are bars arriving? (Check Railway logs for "INCOMING WEBHOOK")
2. Sufficient bars? (Need 30+ bars for FCD warmup)
3. BecomingScore filter? (Set `min_becoming_score=0.0` to disable)

### Issue: Import errors

**Solution:**
```bash
# Ensure all __init__.py files exist
touch fcd/__init__.py
touch fcd/signal/__init__.py
touch fcd/core/__init__.py
touch utils/__init__.py
```

### Issue: Railway deployment fails

**Check:**
1. `requirements.txt` is present
2. `Procfile` contains: `web: uvicorn main:app --host 0.0.0.0 --port $PORT`
3. Python version in `runtime.txt`: `3.11.0`

## 🔒 Security Notes

- This is a **paper trading** system (no real money)
- Webhook endpoint is public but only accepts valid JSON
- No API keys or sensitive data stored
- CSV log is stored in Railway (persists between deploys)

## 📚 API Reference

### POST /webhook

Receives TradingView bar data and processes through FCD engine.

**Request Body:**
```json
{
  "symbol": "SPY",
  "timestamp": 1732561500000,
  "open": 450.25,
  "high": 451.50,
  "low": 449.80,
  "close": 450.90,
  "volume": 65432100
}
```

**Response:**
```json
{
  "status": "success",
  "symbol": "SPY",
  "signal": "LONG",
  "metadata": {
    "fcd_value": 0.6543,
    "becoming_score": 0.1840,
    "confidence": 0.7234,
    "reason": "fcd_long_signal"
  },
  "trade": {
    "action": "BUY",
    "price": 450.90,
    "shares": 210.6512,
    "cash": 5000.00,
    "equity": 100000.00
  },
  "timestamp": "2025-11-25T14:30:00.000Z"
}
```

## 🎓 Next Steps

1. ✅ Deploy to Railway
2. ✅ Configure TradingView alerts
3. ✅ Monitor first signals in Railway logs
4. ✅ Download `trades.csv` to analyze performance
5. ✅ Adjust thresholds based on results
6. 🔜 Add multiple symbols/timeframes
7. 🔜 Implement webhook authentication
8. 🔜 Add email/SMS notifications

## 📞 Support

For questions about:
- **FCD Model:** See original project documentation
- **Railway Deployment:** [Railway Docs](https://docs.railway.app)
- **TradingView Webhooks:** [TradingView Docs](https://www.tradingview.com/support/solutions/43000529348-webhook-alerts/)

---

**Version:** 1.0.0  
**Last Updated:** November 2025  
**Author:** FCD-PSE Project Team
