# 📱 FCD Phone Alert Setup Guide

## Overview

Your FCD system now sends **TWO types of alerts**:

1. **Data Webhook** → Sends bar data to Python server for FCD calculation
2. **Phone Alert** → Receives BUY/SELL/HOLD signal back from Python and sends to your phone

---

## 🚀 Quick Setup (3 Steps)

### Step 1: Create TradingView Alert

1. **Add indicator** `fcd_signal_with_alerts.pine` to your chart
2. Click **⏰ Alert** button (top toolbar)
3. **Condition**: Select "FCD Signal with Phone Alerts"
4. **Settings**:
   - Trigger: **Once Per Bar Close**
   - Expiration: **Open-ended**

### Step 2: Configure Webhook

In the alert dialog:

**Webhook URL**: 
```
https://fcd-indicator-py-production.up.railway.app/webhook
```

**Message**: (Copy exactly)
```
{{plot_0}}
```

### Step 3: Enable Notifications

**Notifications tab**:
- ✅ **Notify on app** (TradingView mobile app)
- ✅ **Push notification**
- ✅ **Play sound**
- ✅ **Send email** (optional)
- ✅ **Show popup**

---

## 📊 What You'll Receive on Your Phone

### Format:
```
🚀 FCD SIGNAL
━━━━━━━━━━━━━━━━━━━━
SYMBOL: MGC=F
SIGNAL: LONG
PRICE: $2043.50
FCD: 0.1247
SCORE: 0.1840
CASH: $95,234.18
TIME: 14:35:22
```

### Signal Types:
- **LONG** = BUY signal (FCD > 0.1, position is FLAT)
- **FLAT** = SELL signal (FCD < -0.1, position is LONG)
- **HOLD** = No action (stay in current position)

---

## 🎛️ How It Works

```
TradingView Chart (bar closes)
    ↓
Sends OHLCV data to Python server
    ↓
Python calculates FCD signal
    ↓
Python sends response with BUY/SELL/HOLD
    ↓
TradingView sends notification to your phone
```

---

## 🧪 Testing Your Setup

### Test with curl:

```bash
curl -X POST https://fcd-indicator-py-production.up.railway.app/webhook \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "SPY",
    "timestamp": 1732579200000,
    "open": 450.25,
    "high": 451.80,
    "low": 449.90,
    "close": 451.20,
    "volume": 1234567
  }'
```

**Expected response:**
```json
{
  "status": "success",
  "symbol": "SPY",
  "signal": "LONG" or "FLAT" or "HOLD",
  "metadata": {
    "fcd_value": 0.1234,
    "becoming_score": 0.0145,
    "confidence": "high"
  },
  "trade": {...}
}
```

---

## ⚙️ Alert Settings in Pine Script

### Inputs (modify in indicator settings):

| Setting | Default | Description |
|---------|---------|-------------|
| `webhook_url` | Railway URL | Your Python server endpoint |
| `ticker_symbol` | SPY | Symbol to trade |
| `enable_alerts` | true | Turn phone alerts on/off |

---

## 🔧 Troubleshooting

### "No alerts received"
1. Check TradingView alert is **active** (green dot)
2. Verify webhook URL is correct
3. Check Railway logs for incoming requests
4. Make sure mobile notifications are enabled in TradingView app

### "Getting data but no signal"
1. Check Railway logs for FCD calculation errors
2. Verify symbol has BecomingScore data
3. Ensure FCD thresholds are appropriate (default: ±0.1)

### "Wrong signal timing"
- Alerts trigger **once per bar close** (default: 1d timeframe)
- Change chart timeframe to get more frequent signals (e.g., 1h, 15m)

### "Phone alerts stopped working"
- TradingView alerts have limits on free plans
- Upgrade to Pro plan for unlimited alerts
- Check alert didn't expire

---

## 📱 Mobile App Setup

1. **Install TradingView app** (iOS/Android)
2. **Login** with your account
3. **Enable notifications**:
   - iOS: Settings → TradingView → Notifications → Allow
   - Android: Settings → Apps → TradingView → Notifications → Enable
4. **Open app** at least once to activate push notifications

---

## 🎯 Recommended Symbols

Based on your BecomingScore rankings:

| Symbol | Score | Description |
|--------|-------|-------------|
| MGC=F | 0.184 | Micro Gold Futures |
| SIL=F | 0.120 | Silver Futures |
| MNQ=F | 0.014 | Micro Nasdaq Futures |
| MES=F | 0.007 | Micro E-mini S&P |

Start with **MGC=F** for best FCD performance.

---

## 🔐 Security Notes

- Your webhook URL is **public** but requires valid JSON format
- No authentication needed (Railway handles HTTPS)
- Rate limiting: ~60 requests/minute per symbol
- Consider adding API key authentication for production

---

## 📊 Monitoring Your Trades

### View live stats:
```bash
curl https://fcd-indicator-py-production.up.railway.app/stats
```

### Response:
```json
{
  "cash": 95234.18,
  "equity": 98567.23,
  "total_trades": 12,
  "winning_trades": 8,
  "losing_trades": 4,
  "win_rate": 66.67,
  "total_pnl": -1432.77
}
```

---

## 🎓 Next Steps

1. ✅ **Setup alert** → Follow Step 1-3 above
2. ✅ **Test with one symbol** → Start with MGC=F
3. ✅ **Monitor for 1 week** → Check phone alerts and Railway logs
4. ✅ **Add more symbols** → Create separate alerts for SIL=F, MNQ=F, etc.
5. ✅ **Optimize thresholds** → Adjust FCD thresholds based on results

---

## 🆘 Support

- **Railway Logs**: https://railway.app/dashboard
- **GitHub Issues**: https://github.com/GabeCabrera/fcd-indicator-py/issues
- **Server Health**: https://fcd-indicator-py-production.up.railway.app/health

---

**🚀 You're all set! Your FCD signals will now arrive on your phone in real-time.**
