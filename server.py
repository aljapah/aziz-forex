#!/usr/bin/env python3
"""
Aziz Forex - Backend Server (BYOK Edition)
"""

from fastapi import FastAPI, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import urllib.request
import json
import os

app = FastAPI()

TIMEFRAME_MAP = {
    "1m": ("1m", "1d"),
    "5m": ("5m", "1d"),
    "15m": ("15m", "5d"),
    "1h": ("1h", "5d"),
    "4h": ("1h", "1mo"),  # Yahoo doesn't support 4h, use 1h
    "1d": ("1d", "3mo"),
}

def get_market_data(symbol: str, timeframe: str = "1h"):
    """جلب بيانات السوق"""
    try:
        interval, range_period = TIMEFRAME_MAP.get(timeframe, ("1h", "5d"))
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval={interval}&range={range_period}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode())
        
        result = data["chart"]["result"][0]
        quotes = result["indicators"]["quote"][0]
        closes = [c for c in quotes["close"] if c is not None]
        
        if not closes:
            return None
        
        price = closes[-1]
        trend = "UP" if len(closes) >= 5 and closes[-1] > closes[-5] else "DOWN"
        
        # RSI
        gains, losses = [], []
        for i in range(1, len(closes)):
            diff = closes[i] - closes[i-1]
            gains.append(diff if diff > 0 else 0)
            losses.append(abs(diff) if diff < 0 else 0)
        
        avg_gain = sum(gains[-14:]) / 14 if len(gains) >= 14 else sum(gains) / max(len(gains), 1)
        avg_loss = sum(losses[-14:]) / 14 if len(losses) >= 14 else sum(losses) / max(len(losses), 1)
        rsi = round(100 - (100 / (1 + avg_gain / max(avg_loss, 0.0001))), 1)
        
        return {"price": round(price, 5), "trend": trend, "rsi": rsi}
        
    except Exception as e:
        print(f"Market data error: {e}")
        return None

STRATEGY_PROMPTS = {
    "rsi": "RSI strategy: BUY if oversold (RSI<30), SELL if overbought (RSI>70).",
    "trend": "Trend Following: BUY in uptrend, SELL in downtrend.",
    "breakout": "Breakout: BUY/SELL on support/resistance breaks.",
    "scalping": "Scalping: Quick trades, tight SL/TP (10-20 pips).",
    "swing": "Swing: Hold days, wider SL/TP based on structure.",
    "price_action": "Price Action: Analyze candlestick patterns only.",
    "smart_money": "Smart Money Concepts: Look for order blocks, liquidity sweeps.",
    "supply_demand": "Supply & Demand: Identify zones and trade reversals."
}

def get_chart_image(symbol: str, timeframe: str = "1h"):
    """جلب صورة الشارت من TradingView"""
    import base64
    
    # تحويل الرمز لصيغة TradingView
    tv_symbol = symbol.replace("-", "").replace("=X", "").replace("^", "")
    if symbol.endswith("-USD"):
        tv_symbol = f"BINANCE:{symbol.replace('-USD', 'USDT')}"
    elif "=X" in symbol:
        tv_symbol = f"FX:{symbol.replace('=X', '')}"
    elif symbol.startswith("^"):
        tv_symbol = f"INDEX:{symbol.replace('^', '')}"
    elif symbol.endswith("=F"):
        tv_symbol = f"COMEX:{symbol.replace('=F', '1!')}"
    else:
        tv_symbol = symbol
    
    # تحويل الإطار الزمني لصيغة TradingView
    tv_interval = {"1m": "1", "5m": "5", "15m": "15", "1h": "60", "4h": "240", "1d": "D"}.get(timeframe, "60")
    
    # استخدام chart-img.com API (مجاني)
    chart_url = f"https://api.chart-img.com/v1/tradingview/advanced-chart?symbol={tv_symbol}&interval={tv_interval}&theme=dark&width=800&height=500&key=demo"
    
    try:
        req = urllib.request.Request(chart_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as response:
            image_data = response.read()
            return base64.b64encode(image_data).decode('utf-8')
    except Exception as e:
        print(f"Chart image error: {e}")
        return None

def analyze_openai(api_key: str, symbol_name: str, price: float, trend: str, rsi: float, strategy: str = "rsi", chart_image: str = None):
    """تحليل باستخدام OpenAI مع صورة الشارت"""
    import openai
    
    client = openai.OpenAI(api_key=api_key)
    
    strategy_hint = STRATEGY_PROMPTS.get(strategy, STRATEGY_PROMPTS["rsi"])
    
    # إذا في صورة، استخدم Vision
    if chart_image:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""أنت محلل فني محترف. حلل هذا الشارت لـ {symbol_name}.

الاستراتيجية: {strategy_hint}
السعر الحالي: {price}
RSI: {rsi}

انظر للشارت وحلل:
- الشموع والأنماط
- الاتجاه العام
- مستويات الدعم والمقاومة
- المؤشرات الظاهرة

إذا وجدت فرصة تداول واضحة، رد بهذا الشكل بالضبط:
🟢 BUY أو 🔴 SELL
{symbol_name}
Entry: السعر
SL: سعر وقف الخسارة
TP: سعر أخذ الربح
Lot: 0.01

إذا لا توجد فرصة واضحة: ⏳

بدون أي شرح إضافي. فقط 6 أسطر أو ⏳"""
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{chart_image}"
                        }
                    }
                ]
            }
        ]
    else:
        # بدون صورة - تحليل بالأرقام فقط
        messages = [
            {
                "role": "user",
                "content": f"""Price: {price}, RSI: {rsi}, Trend: {trend}
Strategy: {strategy_hint}

If opportunity, reply EXACTLY:
🟢 BUY or 🔴 SELL
{symbol_name}
Entry: {price}
SL: number
TP: number
Lot: 0.01

If no opportunity: ⏳

NO TEXT. ONLY 6 LINES OR ⏳"""
            }
        ]
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=100,
        temperature=0.0
    )
    
    return response.choices[0].message.content.strip()

def analyze_anthropic(api_key: str, symbol_name: str, price: float, trend: str, rsi: float, strategy: str = "rsi", chart_image: str = None):
    """تحليل باستخدام Claude مع صورة الشارت"""
    import anthropic
    
    client = anthropic.Anthropic(api_key=api_key)
    strategy_hint = STRATEGY_PROMPTS.get(strategy, STRATEGY_PROMPTS["rsi"])
    
    prompt_text = f"""أنت محلل فني محترف. حلل {symbol_name}.

الاستراتيجية: {strategy_hint}
السعر: {price}, RSI: {rsi}

إذا وجدت فرصة، رد بالضبط:
🟢 BUY أو 🔴 SELL
{symbol_name}
Entry: السعر
SL: وقف الخسارة
TP: أخذ الربح
Lot: 0.01

إذا لا فرصة: ⏳

بدون شرح. 6 أسطر أو ⏳"""

    if chart_image:
        content = [
            {"type": "text", "text": prompt_text},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": chart_image}}
        ]
    else:
        content = prompt_text
    
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=100,
        messages=[{"role": "user", "content": content}]
    )
    
    return response.content[0].text.strip()

def analyze_google(api_key: str, symbol_name: str, price: float, trend: str, rsi: float, strategy: str = "rsi"):
    """تحليل باستخدام Gemini"""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={api_key}"
    strategy_hint = STRATEGY_PROMPTS.get(strategy, STRATEGY_PROMPTS["rsi"])
    
    prompt = f"""Price: {price}, RSI: {rsi}, Trend: {trend}
Strategy: {strategy_hint}

If opportunity based on strategy, reply EXACTLY:
🟢 BUY or 🔴 SELL
{symbol_name}
Entry: {price}
SL: number
TP: number
Lot: 0.01

If no opportunity: ⏳

NO TEXT. ONLY 6 LINES OR ⏳"""

    data = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}]
    }).encode('utf-8')
    
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    
    with urllib.request.urlopen(req, timeout=30) as response:
        result = json.loads(response.read().decode())
    
    return result["candidates"][0]["content"]["parts"][0]["text"].strip()

def send_ntfy(channel: str, title: str, message: str):
    """إرسال إشعار عبر Ntfy"""
    if not channel:
        return
    
    try:
        data = message.encode('utf-8')
        req = urllib.request.Request(
            f"https://ntfy.sh/{channel}",
            data=data,
            headers={"Title": title, "Priority": "high"}
        )
        urllib.request.urlopen(req, timeout=5)
        print(f"Notification sent to {channel}")
    except Exception as e:
        print(f"Ntfy error: {e}")

def send_telegram(chat_id: str, token: str, title: str, message: str):
    """إرسال إشعار عبر Telegram"""
    if not chat_id or not token:
        return
    
    try:
        text = f"*{title}*\n\n{message}"
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        data = json.dumps({
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "Markdown"
        }).encode('utf-8')
        
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=5)
        print(f"Telegram sent to {chat_id}")
    except Exception as e:
        print(f"Telegram error: {e}")

def parse_signal(signal_text: str, price: float):
    """تحليل رد الذكاء الاصطناعي"""
    result = {"signal": signal_text, "price": price}
    
    if "⏳" in signal_text:
        return result
    
    lines = signal_text.strip().split("\n")
    
    for line in lines:
        line = line.strip()
        if "Entry:" in line:
            try:
                result["entry"] = float(line.split(":")[-1].strip())
            except:
                result["entry"] = price
        elif "SL:" in line:
            try:
                result["sl"] = float(line.split(":")[-1].strip())
            except:
                pass
        elif "TP:" in line:
            try:
                result["tp"] = float(line.split(":")[-1].strip())
            except:
                pass
        elif "Lot:" in line:
            try:
                result["lot"] = line.split(":")[-1].strip()
            except:
                result["lot"] = "0.01"
    
    return result

@app.get("/api/analyze")
async def analyze_endpoint(
    symbol: str = Query(...),
    name: str = Query(...),
    apiKey: str = Query(""),
    ai: str = Query("openai"),
    strategy: str = Query("rsi"),
    ntfy: str = Query(""),
    tgChat: str = Query(""),
    tgToken: str = Query(""),
    useChart: str = Query("true"),
    timeframe: str = Query("1h")
):
    """API endpoint للتحليل - BYOK مع صورة الشارت"""
    
    if not apiKey:
        return {"error": "No API key provided", "price": None, "signal": "⏳"}
    
    # جلب البيانات
    data = get_market_data(symbol, timeframe)
    
    if not data:
        return {"error": "Failed to fetch data", "price": None, "signal": "⏳"}
    
    # جلب صورة الشارت
    chart_image = None
    if useChart == "true":
        print(f"Fetching chart for {symbol} ({timeframe})...")
        chart_image = get_chart_image(symbol, timeframe)
        if chart_image:
            print(f"Chart image received ({len(chart_image)} bytes)")
        else:
            print("Chart image failed, using data only")
    
    # تحليل حسب نوع AI
    try:
        if ai == "openai":
            signal_text = analyze_openai(apiKey, name, data["price"], data["trend"], data["rsi"], strategy, chart_image)
        elif ai == "anthropic":
            signal_text = analyze_anthropic(apiKey, name, data["price"], data["trend"], data["rsi"], strategy, chart_image)
        elif ai == "google":
            signal_text = analyze_google(apiKey, name, data["price"], data["trend"], data["rsi"], strategy)
        else:
            signal_text = analyze_openai(apiKey, name, data["price"], data["trend"], data["rsi"], strategy, chart_image)
    except Exception as e:
        print(f"AI Error: {e}")
        return {"error": str(e), "price": data["price"], "signal": "⏳"}
    
    # تحليل الرد
    result = parse_signal(signal_text, data["price"])
    
    # إرسال إشعارات إذا في فرصة
    if "BUY" in signal_text or "SELL" in signal_text:
        signal_type = "🟢 شراء" if "BUY" in signal_text else "🔴 بيع"
        msg = f"{name}\nEntry: {result.get('entry', data['price'])}\nSL: {result.get('sl', '--')}\nTP: {result.get('tp', '--')}\nLot: {result.get('lot', '0.01')}"
        send_ntfy(ntfy, signal_type, msg)
        send_telegram(tgChat, tgToken, signal_type, msg)
    
    return result

@app.get("/")
async def root():
    return FileResponse("index.html")

app.mount("/", StaticFiles(directory="."), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
