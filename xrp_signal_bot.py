import os
import smtplib
import requests
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime, timezone
import pandas as pd
import numpy as np

# ─────────────────────────────────────────
# CONFIG — GitHub Secrets se aata hai
# ─────────────────────────────────────────
GMAIL_USER   = os.environ["GMAIL_USER"]
GMAIL_PASS   = os.environ["GMAIL_PASS"]
NOTIFY_EMAIL = os.environ["NOTIFY_EMAIL"]   # jis email pe alert chahiye

SYMBOL    = "XRPUSDT"
INTERVAL  = "1h"          # 1-hour candles
LIMIT     = 100           # last 100 candles for analysis

# MEXC public API (no key needed for market data)
BASE_URL  = "https://api.mexc.com/api/v3"

# ─────────────────────────────────────────
# STEP 1 — MEXC se XRP candle data fetch
# ─────────────────────────────────────────
def fetch_candles():
    url = f"{BASE_URL}/klines"
    params = {"symbol": SYMBOL, "interval": INTERVAL, "limit": LIMIT}
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    raw = r.json()

    df = pd.DataFrame(raw, columns=[
        "open_time","open","high","low","close","volume",
        "close_time","quote_volume","trades",
        "taker_buy_base","taker_buy_quote","ignore"
    ])
    for col in ["open","high","low","close","volume"]:
        df[col] = df[col].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    return df

# ─────────────────────────────────────────
# STEP 2 — Technical Indicators
# ─────────────────────────────────────────
def add_indicators(df):
    # RSI (14)
    delta = df["close"].diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_g = gain.ewm(com=13, adjust=False).mean()
    avg_l = loss.ewm(com=13, adjust=False).mean()
    rs    = avg_g / avg_l
    df["rsi"] = 100 - (100 / (1 + rs))

    # EMA 20 & EMA 50
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()

    # MACD (12,26,9)
    ema12       = df["close"].ewm(span=12, adjust=False).mean()
    ema26       = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"]  = ema12 - ema26
    df["signal"]= df["macd"].ewm(span=9, adjust=False).mean()
    df["hist"]  = df["macd"] - df["signal"]

    # Bollinger Bands (20, 2σ)
    sma20        = df["close"].rolling(20).mean()
    std20        = df["close"].rolling(20).std()
    df["bb_upper"]= sma20 + 2 * std20
    df["bb_lower"]= sma20 - 2 * std20
    df["bb_mid"]  = sma20

    # ATR (14) — volatility measure
    hl   = df["high"] - df["low"]
    hc   = (df["high"] - df["close"].shift()).abs()
    lc   = (df["low"]  - df["close"].shift()).abs()
    tr   = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()

    # Volume spike (2x average = notable)
    df["vol_avg"]   = df["volume"].rolling(20).mean()
    df["vol_spike"] = df["volume"] > (df["vol_avg"] * 2)

    return df

# ─────────────────────────────────────────
# STEP 3 — Signal Engine
# ─────────────────────────────────────────
def detect_signal(df):
    latest = df.iloc[-1]
    prev   = df.iloc[-2]

    signals = []
    score   = 0   # +ve = LONG bias, -ve = SHORT bias

    # ── LONG conditions ──
    if latest["rsi"] < 35:
        signals.append("✅ RSI oversold (<35) — reversal zone")
        score += 2
    if latest["rsi"] > 50 and prev["rsi"] <= 50:
        signals.append("✅ RSI crossed above 50 — bullish momentum")
        score += 1
    if latest["macd"] > latest["signal"] and prev["macd"] <= prev["signal"]:
        signals.append("✅ MACD bullish crossover")
        score += 2
    if latest["close"] > latest["ema20"] > latest["ema50"]:
        signals.append("✅ Price above EMA20 & EMA50 — uptrend confirmed")
        score += 2
    if latest["close"] <= latest["bb_lower"]:
        signals.append("✅ Price touched lower Bollinger Band — bounce likely")
        score += 2
    if latest["vol_spike"] and latest["close"] > prev["close"]:
        signals.append("✅ Volume spike on green candle — strong buying")
        score += 1

    # ── SHORT conditions ──
    if latest["rsi"] > 70:
        signals.append("🔴 RSI overbought (>70) — reversal zone")
        score -= 2
    if latest["rsi"] < 50 and prev["rsi"] >= 50:
        signals.append("🔴 RSI crossed below 50 — bearish momentum")
        score -= 1
    if latest["macd"] < latest["signal"] and prev["macd"] >= prev["signal"]:
        signals.append("🔴 MACD bearish crossover")
        score -= 2
    if latest["close"] < latest["ema20"] < latest["ema50"]:
        signals.append("🔴 Price below EMA20 & EMA50 — downtrend confirmed")
        score -= 2
    if latest["close"] >= latest["bb_upper"]:
        signals.append("🔴 Price touched upper Bollinger Band — rejection likely")
        score -= 2
    if latest["vol_spike"] and latest["close"] < prev["close"]:
        signals.append("🔴 Volume spike on red candle — strong selling")
        score -= 1

    # ── Candle Pattern (Engulfing) ──
    bull_engulf = (prev["close"] < prev["open"] and
                   latest["close"] > latest["open"] and
                   latest["open"]  < prev["close"] and
                   latest["close"] > prev["open"])
    bear_engulf = (prev["close"] > prev["open"] and
                   latest["close"] < latest["open"] and
                   latest["open"]  > prev["close"] and
                   latest["close"] < prev["open"])

    if bull_engulf:
        signals.append("✅ Bullish Engulfing candle pattern")
        score += 2
    if bear_engulf:
        signals.append("🔴 Bearish Engulfing candle pattern")
        score -= 2

    # ── Decision ──
    if score >= 4:
        direction  = "LONG  📈"
        order_type = "BUY"
        confidence = "HIGH" if score >= 6 else "MEDIUM"
    elif score <= -4:
        direction  = "SHORT 📉"
        order_type = "SELL"
        confidence = "HIGH" if score <= -6 else "MEDIUM"
    else:
        direction  = "NEUTRAL ⏸"
        order_type = "WAIT"
        confidence = "LOW"

    # ── Risk Management ──
    price  = latest["close"]
    atr    = latest["atr"]
    sl_pct = round((atr / price) * 100, 2)        # stop loss = 1 ATR
    tp_pct = round(sl_pct * 2, 2)                  # TP = 2:1 RR ratio

    if order_type == "BUY":
        stop_loss   = round(price - atr, 5)
        take_profit = round(price + (atr * 2), 5)
    elif order_type == "SELL":
        stop_loss   = round(price + atr, 5)
        take_profit = round(price - (atr * 2), 5)
    else:
        stop_loss   = None
        take_profit = None

    return {
        "direction":   direction,
        "order_type":  order_type,
        "confidence":  confidence,
        "score":       score,
        "signals":     signals,
        "price":       price,
        "stop_loss":   stop_loss,
        "take_profit": take_profit,
        "sl_pct":      sl_pct,
        "tp_pct":      tp_pct,
        "rsi":         round(latest["rsi"], 2),
        "macd":        round(latest["macd"], 6),
        "atr":         round(atr, 5),
        "ema20":       round(latest["ema20"], 5),
        "ema50":       round(latest["ema50"], 5),
        "bb_upper":    round(latest["bb_upper"], 5),
        "bb_lower":    round(latest["bb_lower"], 5),
        "candle_time": latest["open_time"].strftime("%Y-%m-%d %H:%M UTC"),
        "vol_spike":   latest["vol_spike"],
    }

# ─────────────────────────────────────────
# STEP 4 — Email Alert
# ─────────────────────────────────────────
def send_email(sig):
    direction  = sig["direction"]
    confidence = sig["confidence"]
    order_type = sig["order_type"]

    subject = f"🚨 XRP/USDT Signal: {order_type} | {confidence} Confidence | {sig['candle_time']}"

    signals_html = "".join(f"<li>{s}</li>" for s in sig["signals"])

    sl_tp_html = ""
    if sig["stop_loss"]:
        sl_tp_html = f"""
        <tr><td><b>Stop Loss</b></td><td>{sig['stop_loss']} USDT &nbsp;(-{sig['sl_pct']}%)</td></tr>
        <tr><td><b>Take Profit</b></td><td>{sig['take_profit']} USDT &nbsp;(+{sig['tp_pct']}%)</td></tr>
        <tr><td><b>Risk:Reward</b></td><td>1 : 2 ✅</td></tr>
        """

    color = {"LONG  📈": "#00c896", "SHORT 📉": "#ff4d6d", "NEUTRAL ⏸": "#888888"}
    bar_color = color.get(direction, "#888888")

    body = f"""
    <html><body style="font-family:monospace;background:#0d0d0d;color:#e0e0e0;padding:24px;">
    <div style="max-width:600px;margin:auto;border:1px solid #222;border-radius:8px;overflow:hidden;">

      <div style="background:{bar_color};padding:16px 24px;">
        <h2 style="margin:0;color:#000;">⚡ XRP/USDT FUTURES SIGNAL</h2>
        <p style="margin:4px 0 0;color:#000;font-size:13px;">MEXC Platform · 1H Timeframe · {sig['candle_time']}</p>
      </div>

      <div style="padding:24px;background:#111;">
        <h3 style="color:{bar_color};font-size:22px;margin:0 0 4px;">
          {direction}
        </h3>
        <p style="color:#aaa;margin:0 0 20px;">Confidence: <b style="color:#fff;">{confidence}</b> &nbsp;|&nbsp; Score: {sig['score']}/10</p>

        <table style="width:100%;border-collapse:collapse;font-size:14px;">
          <tr style="border-bottom:1px solid #222;">
            <td style="padding:8px 0;color:#aaa;width:140px;"><b>Entry Price</b></td>
            <td style="color:#fff;">{sig['price']} USDT</td>
          </tr>
          {sl_tp_html}
          <tr style="border-bottom:1px solid #222;">
            <td style="padding:8px 0;color:#aaa;"><b>RSI (14)</b></td>
            <td style="color:#fff;">{sig['rsi']}</td>
          </tr>
          <tr style="border-bottom:1px solid #222;">
            <td style="padding:8px 0;color:#aaa;"><b>MACD</b></td>
            <td style="color:#fff;">{sig['macd']}</td>
          </tr>
          <tr style="border-bottom:1px solid #222;">
            <td style="padding:8px 0;color:#aaa;"><b>EMA 20 / 50</b></td>
            <td style="color:#fff;">{sig['ema20']} / {sig['ema50']}</td>
          </tr>
          <tr style="border-bottom:1px solid #222;">
            <td style="padding:8px 0;color:#aaa;"><b>BB Upper/Lower</b></td>
            <td style="color:#fff;">{sig['bb_upper']} / {sig['bb_lower']}</td>
          </tr>
          <tr>
            <td style="padding:8px 0;color:#aaa;"><b>Volume Spike</b></td>
            <td style="color:#fff;">{'⚡ YES' if sig['vol_spike'] else 'No'}</td>
          </tr>
        </table>

        <div style="margin-top:20px;background:#1a1a1a;border-left:3px solid {bar_color};padding:12px 16px;border-radius:4px;">
          <p style="margin:0 0 8px;color:#aaa;font-size:12px;text-transform:uppercase;letter-spacing:1px;">Why this signal?</p>
          <ul style="margin:0;padding-left:18px;color:#ddd;font-size:13px;line-height:1.8;">
            {signals_html}
          </ul>
        </div>

        <div style="margin-top:20px;background:#1a1a1a;border-radius:4px;padding:12px 16px;">
          <p style="margin:0;color:#aaa;font-size:12px;text-transform:uppercase;letter-spacing:1px;">⚙️ MEXC Futures Order Setup</p>
          <p style="margin:8px 0 0;color:#ddd;font-size:13px;line-height:1.8;">
            1. MEXC → Futures → <b>XRP/USDT</b><br>
            2. Order Type: <b>Market Order</b> (fast entry)<br>
            3. Direction: <b>{order_type}</b><br>
            4. Leverage: <b>5x–10x max</b> (risk management)<br>
            5. Set SL: <b>{sig['stop_loss']} USDT</b><br>
            6. Set TP: <b>{sig['take_profit']} USDT</b><br>
            7. Use max <b>2–5% of portfolio</b> per trade
          </p>
        </div>

        <p style="margin-top:20px;color:#555;font-size:11px;border-top:1px solid #222;padding-top:12px;">
          ⚠️ Yeh bot technical analysis pe based hai. Koi bhi trade apne risk pe lo.
          Past performance future results guarantee nahi karta. Always use Stop Loss.
        </p>
      </div>
    </div>
    </body></html>
    """

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = GMAIL_USER
    msg["To"]      = NOTIFY_EMAIL
    msg.attach(MIMEText(body, "html"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(GMAIL_USER, GMAIL_PASS)
        server.sendmail(GMAIL_USER, NOTIFY_EMAIL, msg.as_string())

    print(f"✅ Email sent: {subject}")

# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
def main():
    print(f"🔍 Fetching XRP/USDT 1H candles from MEXC... [{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}]")
    df  = fetch_candles()
    df  = add_indicators(df)
    sig = detect_signal(df)

    print(f"📊 Signal: {sig['direction']} | Score: {sig['score']} | Confidence: {sig['confidence']}")
    print(f"   RSI: {sig['rsi']} | MACD: {sig['macd']}")

    # Sirf HIGH ya MEDIUM confidence pe email bhejo
    if sig["order_type"] != "WAIT":
        if sig["confidence"] in ["HIGH", "MEDIUM"]:
            print("📧 Sending email alert...")
            send_email(sig)
        else:
            print("ℹ️  Signal weak — email nahi bheja (LOW confidence)")
    else:
        print("⏸  Market neutral — koi signal nahi mila, wait karo.")

if __name__ == "__main__":
    main()
