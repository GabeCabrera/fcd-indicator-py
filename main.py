from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import uvicorn
import yfinance as yf
import time
from datetime import datetime
import csv
import os
import sys

app = FastAPI()

# Flush logs immediately for Railway visibility
def log_print(msg):
    print(msg)
    sys.stdout.flush()

# ===========================================================
# CORS
# ===========================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    log_print("Invalid JSON received:")
    log_print(str(exc))
    return JSONResponse(status_code=400, content={"error": "Invalid JSON"})


# ===========================================================
# CSV LOGGING
# ===========================================================
CSV_FILE = "trades.csv"

if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp", "action", "symbol", "price", "entry_price",
            "pnl", "cash", "equity", "fcd_score", "confidence"
        ])
    log_print("Created new trades.csv")


def log_trade(action, symbol, price, entry_price, pnl, cash, equity, fcd_score, confidence):
    with open(CSV_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            action,
            symbol,
            price,
            entry_price,
            pnl,
            cash,
            equity,
            fcd_score,
            confidence
        ])

    # Print to Railway logs
    log_print("\n=== TRADE EXECUTED ===")
    log_print(f"Action: {action.upper()}")
    log_print(f"Symbol: {symbol}")
    log_print(f"Price: ${price}")
    log_print(f"Entry Price: ${entry_price}")
    log_print(f"P/L: {pnl}")
    log_print(f"Cash: {cash}")
    log_print(f"Equity: {equity}")
    log_print(f"FCD Score: {fcd_score}")
    log_print(f"Confidence: {confidence * 100:.2f}%")
    log_print("=====================================")


# ===========================================================
# PAPER TRADING ENGINE
# ===========================================================
class PaperAccount:
    def __init__(self):
        self.position = 0
        self.entry_price = 0
        self.cash = 10000
        self.equity = 10000
        self.logs = []

    def buy(self, symbol, price, fcd_score, confidence):
        if price is not None and self.position == 0:
            self.position = 1
            self.entry_price = price

            log_trade("buy", symbol, price, price, 0, self.cash, self.cash, fcd_score, confidence)

    def sell(self, symbol, price, fcd_score, confidence):
        if price is not None and self.position == 1:
            pnl = price - self.entry_price
            self.cash += pnl

            log_trade("sell", symbol, price, self.entry_price, pnl, self.cash, self.cash, fcd_score, confidence)

            self.position = 0

    def status(self):
        return {
            "position": self.position,
            "entry_price": self.entry_price,
            "cash": self.cash,
            "logs": self.logs[-10:]
        }

account = PaperAccount()


# ===========================================================
# WEBHOOK ENDPOINT
# ===========================================================
@app.post("/webhook")
async def webhook(request: Request):
    log_print("\n\n========== Webhook Received ==========")

    try:
        payload = await request.json()
    except Exception as e:
        log_print(f"Failed to parse JSON: {e}")
        return {"error": "Invalid JSON"}

    symbol = payload.get("symbol", "SPY")
    signal = payload.get("signal", "flat")
    fcd_score = payload.get("fcd_score", 0.0)
    confidence = payload.get("confidence", 0.0)

    log_print(f"Time: {datetime.now()}")
    log_print(f"Symbol: {symbol}")
    log_print(f"Signal: {signal}")
    log_print(f"FCD Score: {fcd_score}")
    log_print(f"Confidence: {confidence * 100:.2f}%")

    # =======================================================
    # Fetch yfinance 15m data
    # =======================================================
    try:
        df = yf.download(symbol, interval="15m", period="1d", threads=False)
        if df is None or len(df) == 0:
            last_price = None
        else:
            last_price = float(df["Close"].iloc[-1])
    except Exception as e:
        log_print(f"YFinance Error: {e}")
        last_price = None

    log_print(f"Last SPY 15m Price: {last_price}")

    # Execute logic
    if signal == "buy":
        account.buy(symbol, last_price, fcd_score, confidence)
    elif signal == "sell":
        account.sell(symbol, last_price, fcd_score, confidence)
    else:
        log_print("No trade executed (flat signal).")

    log_print("========================================\n")

    return {
        "status": "ok",
        "received": payload,
        "last_price": last_price,
        "account": account.status()
    }


# ===========================================================
# HEALTH CHECK
# ===========================================================
@app.get("/")
def home():
    return {"status": "running"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
