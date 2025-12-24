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

def get_market_data(symbol: str):
    """جلب بيانات السوق"""
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=15m&range=1d"
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

def analyze_openai(api_key: str, symbol_name: str, price: float, trend: str, rsi: float):
    """تحليل باستخدام OpenAI"""
    import openai
    
    client = openai.OpenAI(api_key=api_key)
    
    prompt = f"""Price: {price}, RSI: {rsi}, Trend: {trend}

If opportunity, reply EXACTLY (6 lines):
🟢 BUY or 🔴 SELL
{symbol_name}
Entry: {price}
SL: number
TP: number
Lot: 0.01

If no opportunity: ⏳

NO TEXT. ONLY 6 LINES OR ⏳"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=50,
        temperature=0.0
    )
    
    return response.choices[0].message.content.strip()

def analyze_anthropic(api_key: str, symbol_name: str, price: float, trend: str, rsi: float):
    """تحليل باستخدام Claude"""
    import anthropic
    
    client = anthropic.Anthropic(api_key=api_key)
    
    prompt = f"""Price: {price}, RSI: {rsi}, Trend: {trend}

If opportunity, reply EXACTLY (6 lines):
🟢 BUY or 🔴 SELL
{symbol_name}
Entry: {price}
SL: number
TP: number
Lot: 0.01

If no opportunity: ⏳

NO TEXT. ONLY 6 LINES OR ⏳"""

    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=50,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.content[0].text.strip()

def analyze_google(api_key: str, symbol_name: str, price: float, trend: str, rsi: float):
    """تحليل باستخدام Gemini"""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={api_key}"
    
    prompt = f"""Price: {price}, RSI: {rsi}, Trend: {trend}

If opportunity, reply EXACTLY (6 lines):
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
    ntfy: str = Query("")
):
    """API endpoint للتحليل - BYOK"""
    
    if not apiKey:
        return {"error": "No API key provided", "price": None, "signal": "⏳"}
    
    # جلب البيانات
    data = get_market_data(symbol)
    
    if not data:
        return {"error": "Failed to fetch data", "price": None, "signal": "⏳"}
    
    # تحليل حسب نوع AI
    try:
        if ai == "openai":
            signal_text = analyze_openai(apiKey, name, data["price"], data["trend"], data["rsi"])
        elif ai == "anthropic":
            signal_text = analyze_anthropic(apiKey, name, data["price"], data["trend"], data["rsi"])
        elif ai == "google":
            signal_text = analyze_google(apiKey, name, data["price"], data["trend"], data["rsi"])
        else:
            signal_text = analyze_openai(apiKey, name, data["price"], data["trend"], data["rsi"])
    except Exception as e:
        print(f"AI Error: {e}")
        return {"error": str(e), "price": data["price"], "signal": "⏳"}
    
    # تحليل الرد
    result = parse_signal(signal_text, data["price"])
    
    # إرسال إشعار إذا في فرصة
    if "BUY" in signal_text or "SELL" in signal_text:
        signal_type = "🟢 شراء" if "BUY" in signal_text else "🔴 بيع"
        msg = f"{name}\nEntry: {result.get('entry', data['price'])}\nSL: {result.get('sl', '--')}\nTP: {result.get('tp', '--')}\nLot: {result.get('lot', '0.01')}"
        send_ntfy(ntfy, signal_type, msg)
    
    return result

@app.get("/")
async def root():
    return FileResponse("index.html")

app.mount("/", StaticFiles(directory="."), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
