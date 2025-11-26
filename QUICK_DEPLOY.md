# 🚀 Quick Deploy to Railway

## Prerequisites
- GitHub account
- Railway account (free tier: railway.app)

## Deploy in 5 Minutes

### 1️⃣ Push to GitHub

```bash
cd fcd-cloud-server

# Initialize git if needed
git init

# Add all files
git add .

# Commit
git commit -m "FCD Cloud Trading Engine - Ready for Railway"

# Create new repo on GitHub: fcd-cloud-server
# Then push:
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/fcd-cloud-server.git
git push -u origin main
```

### 2️⃣ Deploy on Railway

1. Go to https://railway.app
2. Sign in with GitHub
3. Click **"New Project"**
4. Select **"Deploy from GitHub repo"**
5. Choose `fcd-cloud-server`
6. Railway auto-deploys (takes 2-3 minutes)

### 3️⃣ Get Your Webhook URL

After deployment:
1. Click on your service
2. Click **"Settings"**
3. Scroll to **"Domains"**
4. Click **"Generate Domain"**
5. Copy the URL: `https://fcd-cloud-server-production-XXXX.up.railway.app`

Your webhook endpoint is: `https://YOUR-URL/webhook`

### 4️⃣ Test It

```bash
# Test health endpoint
curl https://YOUR-RAILWAY-URL/health

# Test webhook (simulate TradingView)
curl -X POST https://YOUR-RAILWAY-URL/webhook \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "MGC=F",
    "timestamp": 1732561500000,
    "open": 2450.25,
    "high": 2455.80,
    "low": 2448.90,
    "close": 2454.50,
    "volume": 12450
  }'
```

### 5️⃣ Configure TradingView

1. Open TradingView chart (MGC, SPY, etc.)
2. Click 🔔 Alert button
3. **Condition:** Bar Close
4. **Options:** Webhook URL

**Webhook URL:**
```
https://YOUR-RAILWAY-URL/webhook
```

**Message:**
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

5. **Frequency:** Every Time
6. Click **Create**

### 6️⃣ Monitor Logs

Railway Dashboard → Click your service → **"View Logs"**

You'll see:
```
📊 INCOMING WEBHOOK
Symbol: MGC=F
🔬 FCD ANALYSIS
📊 FCD OUTPUT: LONG
🟢 EXECUTING BUY ORDER
💰 TRADE EXECUTED
```

## ✅ Done!

Your FCD engine is now running 24/7 on Railway, processing TradingView alerts!

## 📊 View Results

Download trades.csv from Railway or use the stats endpoint:

```bash
curl https://YOUR-RAILWAY-URL/stats
```

## 🔧 Update Rankings

To update BecomingScore rankings:

1. Run your local ranking script
2. Copy new `consolidated_futures.csv` to `fcd/rankings/`
3. Commit and push:
   ```bash
   git add fcd/rankings/consolidated_futures.csv
   git commit -m "Update rankings"
   git push
   ```
4. Railway auto-redeploys

## 💡 Tips

- **Free tier:** 500 hours/month (plenty for 24/7)
- **Logs:** Keep last 7 days
- **Persistence:** trades.csv persists between deploys
- **Scaling:** Upgrade plan for multiple instances

## ⚠️ Troubleshooting

**No logs appearing?**
- Check TradingView alert is firing
- Verify webhook URL is correct
- Test with curl command above

**Import errors?**
- Railway auto-installs from requirements.txt
- Check deployment logs for errors

**Out of memory?**
- Default 512MB should be fine
- Upgrade if processing many symbols

---

**Need help?** Check full README.md for detailed docs.
