from fastapi import FastAPI, Request
import uvicorn
import yfinance as yf
import time
import json
from datetime import datetime

app = FastAPI()


# ===========================================================
# PAPER TRADING ACCOUNT
# ===========================================================

class PaperAccount:
    def __init__(self):
        self.position = 0
        self.entry_price = 0
        self.cash = 10000
        self.equity = 10000
        self.logs = []

    def buy(self, price):
        if self.position == 0:
            self.position = 1
            self.entry_price = price
            self.logs.append(f"[{datetime.now()}] BUY at ${price}")
            print(self.logs[-1])

    def sell(self, price):
        if self.position == 1:
            pnl = price - self.entry_price
            self.cash += pnl
            self.position = 0
            self.logs.append(f"[{datetime.now()}] SELL at ${price} (P/L: {pnl})")
            print(self.logs[-1])

    def status(self):
        return {
            "position": self.position,
            "entry_price": self.entry_price,
            "cash": self.cash,
            "logs": self.logs[-10:]
        }


account = PaperAccount()


# ===========================================================
# HANDLE TRADINGVIEW WEBHOOK
# ===========================================================
@app.post("/webhook")
async def webhook(request: Request):
    payload = await request.json()

    symbol = payload.get("symbol", "SPY")
    signal = payload.get("signal", "flat")
    fcd_score = payload.get("fcd_score", 0.0)
    confidence = payload.get("confidence", 0.0)

    print("\n=== Webhook Received ===")
    print(payload)

    # =======================================================
    # Fetch SPY real-time 15-minute bar
    # =======================================================
    df = yf.download(symbol, interval="15m", period="1d")
    last_price = float(df["Close"].iloc[-1])

    print(f"Last 15m price for {symbol} = ${last_price}")

    # =======================================================
    # Execute paper trade based on signal
    # =======================================================
    if signal == "buy":
        account.buy(last_price)

    elif signal == "sell":
        account.sell(last_price)

    else:
        print("Flat — no trade.")

    # =======================================================
    # Return local account status
    # =======================================================
    return {
        "received": payload,
        "last_price": last_price,
        "paper_account": account.status()
    }


# ===========================================================
# HEALTH CHECK
# ===========================================================
@app.get("/")
def home():
    return {"status": "running", "time": time.time()}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
