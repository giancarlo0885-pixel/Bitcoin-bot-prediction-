import math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

try:
    import yfinance as yf
except Exception:
    yf = None

try:
    import requests
except Exception:
    requests = None

st.set_page_config(
    page_title="GARIBALDI CRYPTO PREDICTION BOT™",
    page_icon="🤖",
    layout="wide",
)

DEFAULT_CRYPTOS = [
    "BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "DOGE-USD",
    "ADA-USD", "AVAX-USD", "LINK-USD", "LTC-USD", "BCH-USD",
    "XLM-USD", "DOT-USD", "ETC-USD", "XMR-USD"
]

def clean_tickers(raw):
    return [x.strip().upper() for x in raw.replace("\n", ",").split(",") if x.strip()]

def fmt(v):
    if v is None or pd.isna(v):
        return None
    v = float(v)
    return round(v, 5) if abs(v) < 10 else round(v, 2)

@st.cache_data(ttl=180)
def load_prices(tickers, period="6mo", interval="1h"):
    if yf is None:
        return {}
    out = {}
    for tk in tickers:
        try:
            df = yf.Ticker(tk).history(period=period, interval=interval, auto_adjust=True)
            if not df.empty:
                out[tk] = df.reset_index()
        except Exception:
            pass
    return out

@st.cache_data(ttl=300)
def fetch_fear_greed():
    if requests is None:
        return None
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=1", timeout=6)
        item = r.json()["data"][0]
        return {"value": int(item["value"]), "classification": item["value_classification"]}
    except Exception:
        return None

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def atr(df, period=14):
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([(high-low), (high-prev_close).abs(), (low-prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def add_indicators(df):
    df = df.copy()
    close = df["Close"].astype(float)
    df["EMA9"] = close.ewm(span=9, adjust=False).mean()
    df["EMA21"] = close.ewm(span=21, adjust=False).mean()
    df["EMA50"] = close.ewm(span=50, adjust=False).mean()
    df["RSI"] = rsi(close)
    df["ATR"] = atr(df)
    df["RET"] = close.pct_change()
    df["VOLATILITY"] = df["RET"].rolling(30).std() * math.sqrt(365)
    df["VOLUME_MA"] = df["Volume"].rolling(20).mean() if "Volume" in df.columns else 0
    df["VOLUME_SPIKE"] = df["Volume"] / df["VOLUME_MA"] if "Volume" in df.columns else 0
    return df

def prediction_signal(df):
    df = add_indicators(df)
    if len(df) < 60:
        return {"Action": "WAIT", "Confidence": 0, "Reason": "Not enough data"}

    last = df.iloc[-1]
    prev = df.iloc[-2]
    close = float(last["Close"])
    score = 0
    reasons = []

    if last["EMA9"] > last["EMA21"] > last["EMA50"]:
        score += 3
        reasons.append("bullish EMA stack")
    elif last["EMA9"] < last["EMA21"] < last["EMA50"]:
        score -= 3
        reasons.append("bearish EMA stack")

    if prev["EMA9"] <= prev["EMA21"] and last["EMA9"] > last["EMA21"]:
        score += 3
        reasons.append("fresh bullish EMA cross")
    elif prev["EMA9"] >= prev["EMA21"] and last["EMA9"] < last["EMA21"]:
        score -= 3
        reasons.append("fresh bearish EMA cross")

    if 45 <= last["RSI"] <= 68:
        score += 2
        reasons.append("RSI in healthy momentum zone")
    elif last["RSI"] > 78:
        score -= 2
        reasons.append("RSI overheated")
    elif last["RSI"] < 30:
        score += 1
        reasons.append("RSI oversold bounce zone")

    mom_12 = (df["Close"].iloc[-1] / df["Close"].iloc[-12] - 1) * 100 if len(df) >= 12 else 0
    mom_48 = (df["Close"].iloc[-1] / df["Close"].iloc[-48] - 1) * 100 if len(df) >= 48 else 0

    if mom_12 > 2 and mom_48 > 4:
        score += 2
        reasons.append("short and medium momentum aligned")
    elif mom_12 < -2 and mom_48 < -4:
        score -= 2
        reasons.append("downside momentum aligned")

    if last["VOLUME_SPIKE"] and last["VOLUME_SPIKE"] > 1.7 and mom_12 > 0:
        score += 2
        reasons.append("bullish volume spike")
    elif last["VOLUME_SPIKE"] and last["VOLUME_SPIKE"] > 1.7 and mom_12 < 0:
        score -= 2
        reasons.append("bearish volume spike")

    confidence = min(95, max(5, 50 + score * 6))
    atr_value = float(last["ATR"]) if not pd.isna(last["ATR"]) and last["ATR"] > 0 else close * 0.03

    if score >= 6:
        action = "BUY / LONG WATCH"
        stop = close - atr_value * 1.5
        take = close + atr_value * 3
    elif score <= -6:
        action = "SELL / SHORT WATCH"
        stop = close + atr_value * 1.5
        take = close - atr_value * 3
    else:
        action = "WAIT"
        stop = None
        take = None

    return {
        "Action": action,
        "Confidence": round(confidence, 1),
        "Score": score,
        "Price": fmt(close),
        "Stop Loss": fmt(stop) if stop else None,
        "Take Profit": fmt(take) if take else None,
        "RSI": round(float(last["RSI"]), 1) if not pd.isna(last["RSI"]) else None,
        "Volume Spike": round(float(last["VOLUME_SPIKE"]), 2) if not pd.isna(last["VOLUME_SPIKE"]) else None,
        "Reason": ", ".join(reasons) if reasons else "No strong edge",
    }

def backtest_strategy(df, starting_cash=1000, risk_pct=2.0, fee_pct=0.10):
    df = add_indicators(df).dropna().reset_index(drop=True)
    if len(df) < 80:
        return pd.DataFrame(), {"Final Equity": starting_cash, "Total Return %": 0, "Trades": 0, "Win Rate %": 0}

    cash = starting_cash
    position = 0
    entry_price = 0
    stop = None
    take = None
    trades = []
    equity_curve = []

    for i in range(60, len(df)):
        window = df.iloc[:i+1].copy()
        sig = prediction_signal(window)
        price = float(df["Close"].iloc[i])
        time_value = df.iloc[i][0]

        if position > 0:
            exit_reason = None
            if price <= stop:
                exit_reason = "Stop Loss"
            elif price >= take:
                exit_reason = "Take Profit"
            elif sig["Action"] == "SELL / SHORT WATCH":
                exit_reason = "Signal Exit"

            if exit_reason:
                gross = position * price
                fee = gross * fee_pct / 100
                cash = gross - fee
                pnl = cash - starting_cash if len(trades) == 0 else cash - trades[-1].get("Equity Before", starting_cash)
                trades.append({
                    "Time": str(time_value),
                    "Type": "EXIT",
                    "Price": fmt(price),
                    "Reason": exit_reason,
                    "Cash": round(cash, 2),
                    "PnL": round(pnl, 2),
                })
                position = 0
                entry_price = 0
                stop = None
                take = None

        if position == 0 and sig["Action"] == "BUY / LONG WATCH" and sig["Confidence"] >= 62:
            risk_cash = cash * (risk_pct / 100)
            stop_price = sig["Stop Loss"]
            if stop_price and price > stop_price:
                qty_by_risk = risk_cash / (price - stop_price)
                qty_by_cash = cash / price
                qty = min(qty_by_risk, qty_by_cash)
                cost = qty * price
                fee = cost * fee_pct / 100
                if qty > 0 and cost + fee <= cash:
                    equity_before = cash
                    cash -= cost + fee
                    position = qty
                    entry_price = price
                    stop = stop_price
                    take = sig["Take Profit"]
                    trades.append({
                        "Time": str(time_value),
                        "Type": "ENTRY",
                        "Price": fmt(price),
                        "Reason": sig["Reason"],
                        "Cash": round(cash, 2),
                        "Equity Before": round(equity_before, 2),
                    })

        equity = cash + position * price
        equity_curve.append({"Time": str(time_value), "Equity": equity})

    final_price = float(df["Close"].iloc[-1])
    final_equity = cash + position * final_price
    exits = [t for t in trades if t["Type"] == "EXIT"]
    wins = [t for t in exits if t.get("PnL", 0) > 0]
    summary = {
        "Final Equity": round(final_equity, 2),
        "Total Return %": round((final_equity / starting_cash - 1) * 100, 2),
        "Trades": len(exits),
        "Win Rate %": round(len(wins) / len(exits) * 100, 1) if exits else 0,
    }
    return pd.DataFrame(trades), summary

def position_sizing(account_size, risk_pct, entry, stop):
    if not stop or entry <= stop:
        return {"Position Size $": 0, "Coin Qty": 0, "Risk $": 0}
    risk_dollars = account_size * risk_pct / 100
    qty = risk_dollars / (entry - stop)
    size = qty * entry
    return {
        "Position Size $": round(size, 2),
        "Coin Qty": round(qty, 8),
        "Risk $": round(risk_dollars, 2),
    }

def chart(df, ticker):
    df = add_indicators(df)
    xcol = "Datetime" if "Datetime" in df.columns else "Date" if "Date" in df.columns else df.columns[0]
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df[xcol], open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name=ticker))
    fig.add_trace(go.Scatter(x=df[xcol], y=df["EMA9"], mode="lines", name="EMA9"))
    fig.add_trace(go.Scatter(x=df[xcol], y=df["EMA21"], mode="lines", name="EMA21"))
    fig.add_trace(go.Scatter(x=df[xcol], y=df["EMA50"], mode="lines", name="EMA50"))
    fig.update_layout(height=460, title=f"{ticker} Prediction Bot Chart", margin=dict(l=10, r=10, t=35, b=10), xaxis_rangeslider_visible=False)
    return fig


def build_training_timeline(df, ticker, account_size=1000, risk_pct=2.0):
    """
    Hypothetical training body: creates a live-action style replay of how the bot would think.
    No real orders are placed. This is for learning trade execution discipline.
    """
    df = add_indicators(df).dropna().reset_index(drop=True)
    if len(df) < 90:
        return pd.DataFrame(), "Not enough data for training replay."

    events = []
    virtual_cash = account_size
    position = 0
    entry = None
    stop = None
    take = None
    training_score = 100

    start_index = max(60, len(df) - 80)

    for i in range(start_index, len(df)):
        window = df.iloc[:i+1].copy()
        sig = prediction_signal(window)
        price = float(df["Close"].iloc[i])
        time_value = str(df.iloc[i][0])

        if position == 0:
            if sig["Action"] == "BUY / LONG WATCH" and sig["Confidence"] >= 62:
                sizing = position_sizing(virtual_cash, risk_pct, price, sig.get("Stop Loss"))
                if sizing["Coin Qty"] > 0:
                    position = sizing["Coin Qty"]
                    entry = price
                    stop = sig["Stop Loss"]
                    take = sig["Take Profit"]
                    virtual_cash -= sizing["Position Size $"]
                    events.append({
                        "Time": time_value,
                        "Scene": "ENTRY",
                        "Bot Action": "Hypothetical BUY",
                        "Price": fmt(price),
                        "Training Narration": f"Bot sees momentum confirmation: {sig['Reason']}. It enters small and defines risk first.",
                        "Stop": stop,
                        "Take Profit": take,
                        "Lesson": "Never enter without a stop-loss and position size.",
                    })
                else:
                    events.append({
                        "Time": time_value,
                        "Scene": "WAIT",
                        "Bot Action": "No Trade",
                        "Price": fmt(price),
                        "Training Narration": "Signal appeared, but risk sizing was not valid.",
                        "Stop": None,
                        "Take Profit": None,
                        "Lesson": "Bad risk math means no trade.",
                    })
            else:
                if i % 10 == 0:
                    events.append({
                        "Time": time_value,
                        "Scene": "WATCH",
                        "Bot Action": "Scanning",
                        "Price": fmt(price),
                        "Training Narration": f"Bot waits. Current read: {sig['Action']} at {sig['Confidence']}% confidence.",
                        "Stop": None,
                        "Take Profit": None,
                        "Lesson": "Patience is part of the strategy.",
                    })
        else:
            if price <= stop:
                pnl = (price - entry) * position
                training_score -= 10
                events.append({
                    "Time": time_value,
                    "Scene": "STOP HIT",
                    "Bot Action": "Exit Trade",
                    "Price": fmt(price),
                    "Training Narration": f"Price hit stop. Bot exits instead of hoping. Hypothetical P/L: ${round(pnl, 2)}.",
                    "Stop": stop,
                    "Take Profit": take,
                    "Lesson": "Small controlled losses keep the account alive.",
                })
                virtual_cash += position * price
                position = 0
                entry = None
                stop = None
                take = None
            elif price >= take:
                pnl = (price - entry) * position
                training_score += 8
                events.append({
                    "Time": time_value,
                    "Scene": "TAKE PROFIT",
                    "Bot Action": "Exit Winner",
                    "Price": fmt(price),
                    "Training Narration": f"Target hit. Bot takes profit instead of getting greedy. Hypothetical P/L: ${round(pnl, 2)}.",
                    "Stop": stop,
                    "Take Profit": take,
                    "Lesson": "Profit is only real when risk is closed or reduced.",
                })
                virtual_cash += position * price
                position = 0
                entry = None
                stop = None
                take = None
            elif sig["Action"] == "SELL / SHORT WATCH":
                pnl = (price - entry) * position
                events.append({
                    "Time": time_value,
                    "Scene": "SIGNAL EXIT",
                    "Bot Action": "Exit Trade",
                    "Price": fmt(price),
                    "Training Narration": f"Momentum flipped against the trade. Bot exits. Hypothetical P/L: ${round(pnl, 2)}.",
                    "Stop": stop,
                    "Take Profit": take,
                    "Lesson": "When the reason for the trade disappears, exit.",
                })
                virtual_cash += position * price
                position = 0
                entry = None
                stop = None
                take = None

    result = pd.DataFrame(events)
    summary = f"Training Score: {max(0, min(100, training_score))}/100. Replay teaches entries, patience, risk control, stop discipline, and profit-taking."
    return result, summary

def coach_feedback(signal, backtest_summary):
    """
    Converts bot data into a coach-style lesson.
    """
    feedback = []
    if signal["Action"] == "BUY / LONG WATCH":
        feedback.append("Coach: This is a possible long setup, but only valid if the stop-loss and position size fit your account.")
    elif signal["Action"] == "SELL / SHORT WATCH":
        feedback.append("Coach: Momentum is weak. For most beginners, this is a warning to avoid buying, not a reason to gamble short.")
    else:
        feedback.append("Coach: Waiting is a position. No trade is better than a forced trade.")

    if backtest_summary["Trades"] == 0:
        feedback.append("Coach: The strategy did not find enough clean trades in this window. Do not force it.")
    elif backtest_summary["Win Rate %"] < 40:
        feedback.append("Coach: Win rate is weak. Lower risk, adjust settings, or avoid this coin right now.")
    elif backtest_summary["Total Return %"] > 0:
        feedback.append("Coach: Backtest is positive, but still paper trade before risking real money.")

    feedback.append("Coach: The goal is not to be right every trade. The goal is to survive long enough for good setups to pay.")
    return "\n".join([f"- {x}" for x in feedback])

st.title("GARIBALDI CRYPTO PREDICTION BOT™ v3")
st.caption("Crypto prediction signals + paper trading + live-action training simulator.")

with st.sidebar:
    st.header("Bot Controls")
    crypto_input = st.text_area("Crypto tickers", value=", ".join(DEFAULT_CRYPTOS), height=130)
    period = st.selectbox("History", ["1mo", "3mo", "6mo", "1y", "2y"], index=2)
    interval = st.selectbox("Interval", ["1h", "30m", "15m", "1d"], index=0)
    account_size = st.number_input("Account size $", value=1000.0, min_value=10.0)
    risk_pct = st.slider("Risk per trade %", 0.25, 10.0, 2.0)
    fee_pct = st.number_input("Trading fee %", value=0.10, min_value=0.0, step=0.01)
    st.warning("Paper trading only. Do not connect real money until tested.")

tickers = clean_tickers(crypto_input)
prices = load_prices(tickers, period=period, interval=interval)
if not prices:
    st.error("No crypto data loaded.")
    st.stop()

rows = []
signals = {}
for tk, df in prices.items():
    sig = prediction_signal(df)
    signals[tk] = sig
    sizing = position_sizing(account_size, risk_pct, float(df["Close"].iloc[-1]), sig.get("Stop Loss"))
    rows.append({
        "Ticker": tk,
        "Action": sig["Action"],
        "Confidence": sig["Confidence"],
        "Score": sig.get("Score"),
        "Price": sig["Price"],
        "Stop Loss": sig["Stop Loss"],
        "Take Profit": sig["Take Profit"],
        "Position Size $": sizing["Position Size $"],
        "Risk $": sizing["Risk $"],
        "RSI": sig["RSI"],
        "Volume Spike": sig["Volume Spike"],
        "Reason": sig["Reason"],
    })

scan = pd.DataFrame(rows).sort_values(["Confidence", "Score"], ascending=False)

fear = fetch_fear_greed()
buy_count = (scan["Action"] == "BUY / LONG WATCH").sum()
sell_count = (scan["Action"] == "SELL / SHORT WATCH").sum()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Coins Scanned", len(scan))
c2.metric("Buy Watches", int(buy_count))
c3.metric("Sell Watches", int(sell_count))
c4.metric("Fear & Greed", fear["value"] if fear else "N/A", fear["classification"] if fear else "Offline")

st.subheader("Prediction Bot Signal Board")
st.caption("The bot looks for EMA trend, EMA cross, RSI zone, momentum alignment, volume anomaly, ATR stop, and risk-based position sizing.")
st.dataframe(scan, use_container_width=True, hide_index=True)

selected = st.selectbox("Analyze / backtest coin", list(prices.keys()))
df = prices[selected]
sig = signals[selected]

left, right = st.columns([2, 1])
with left:
    st.plotly_chart(chart(df, selected), use_container_width=True)
with right:
    st.subheader(f"{selected} Trade Plan")
    st.metric("Action", sig["Action"], f"{sig['Confidence']}% confidence")
    st.write(f"**Entry area:** ${sig['Price']}")
    st.write(f"**Stop loss:** ${sig['Stop Loss']}")
    st.write(f"**Take profit:** ${sig['Take Profit']}")
    sizing = position_sizing(account_size, risk_pct, float(df["Close"].iloc[-1]), sig.get("Stop Loss"))
    st.write(f"**Suggested position:** ${sizing['Position Size $']}")
    st.write(f"**Coin qty:** {sizing['Coin Qty']}")
    st.write(f"**Risk:** ${sizing['Risk $']}")
    st.write(f"**Why:** {sig['Reason']}")

st.subheader("Paper Trading Backtest")
trades, summary = backtest_strategy(df, starting_cash=account_size, risk_pct=risk_pct, fee_pct=fee_pct)

b1, b2, b3, b4 = st.columns(4)
b1.metric("Final Equity", f"${summary['Final Equity']}")
b2.metric("Return", f"{summary['Total Return %']}%")
b3.metric("Closed Trades", summary["Trades"])
b4.metric("Win Rate", f"{summary['Win Rate %']}%")

if not trades.empty:
    st.dataframe(trades.tail(50), use_container_width=True, hide_index=True)
else:
    st.info("No backtest trades triggered with current settings.")


st.subheader("Live-Action Training Simulator")
st.caption("Hypothetical replay only. The system acts like a training coach showing how the bot would trade step-by-step without real money.")

training_events, training_summary = build_training_timeline(df, selected, account_size=account_size, risk_pct=risk_pct)
st.info(training_summary)

if not training_events.empty:
    scene_filter = st.selectbox("Training scene filter", ["All"] + sorted(training_events["Scene"].unique().tolist()))
    display_events = training_events if scene_filter == "All" else training_events[training_events["Scene"] == scene_filter]
    st.dataframe(display_events.tail(60), use_container_width=True, hide_index=True)
else:
    st.warning("No training events generated for this coin/window.")

st.subheader("Bot Coach")
st.markdown(coach_feedback(sig, summary))

st.subheader("Bot Safety Rules")
st.markdown("""
- Paper trade first.
- Risk 1–2% per trade until proven.
- Never average down blindly.
- Avoid trades when confidence is low.
- Avoid trading during extreme news unless that is part of the tested strategy.
- Real auto-trading should require exchange API keys, max daily loss, emergency kill switch, and manual approval.
""")
