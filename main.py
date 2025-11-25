from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import uvicorn
import yfinance as yf
import time
from datetime import datetime

app = FastAPI()

# ===========================================================
# CORS (TradingView → Railway fix)
# ===========================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===========================================================
# ERROR HANDLER (TradingView invalid JSON fix)
# ===========================================================
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    print("Invalid JSON received:", exc)
    return JSONResponse(status_code=400, content={"error": "Invalid JSON"})


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

    def buy(self, price):
        if price is not None and self.position == 0:
            self.position = 1
            self.entry_price = price
            self.logs.append(f"[{datetime.now()}] BUY at ${price}")
            print(self.logs[-1])

    def sell(self, price):
        if price is not None and self.position == 1:
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
    print("\n=== Incoming request to /webhook ===")

    try:
        payload = await request.json()
    except Exception as e:
        print("JSON parse error:", e)
        return {"error": "Invalid JSON"}

    symbol = payload.get("symbol", "SPY")
    signal = payload.get("signal", "flat")
    fcd_score = payload.get("fcd_score", 0.0)
    confidence = payload.get("confidence", 0.0)

    print("Payload received:", payload)

    # Fetch 15m SPY data (safe)
    try:
        df = yf.download(symbol, interval="15m", period="1d", threads=False)
        if df is None or len(df) == 0:
            print("No data from yfinance.")
            last_price = None
        else:
            last_price = float(df["Close"].iloc[-1])
    except Exception as e:
        print("yfinance error:", e)
        last_price = None

    print(f"Last price for {symbol}: {last_price}")

    # Trading logic
    if signal == "buy":
        account.buy(last_price)
    elif signal == "sell":
        account.sell(last_price)
    else:
        print("Flat — no trade executed.")

    return {
        "received": payload,
        "last_price": last_price,
        "account": account.status()
    }


# ===========================================================
# HEALTH CHECK
# ===========================================================
@app.get("/")
def home():
    return {"status": "running", "time": time.time()}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
