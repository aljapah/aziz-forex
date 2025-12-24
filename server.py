#!/usr/bin/env python3
"""
Aziz Forex - Backend Server
"""

from fastapi import FastAPI, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import urllib.request
import json
import openai

app = FastAPI()

# Ntfy
NTFY_TOPIC = "azizforex2024"

def send_notification(title: str, message: str):
    """إرسال إشعار عبر Ntfy"""
    try:
        data = message.encode('utf-8')
        req = urllib.request.Request(
            f"https://ntfy.sh/{NTFY_TOPIC}",
            data=data,
            headers={
                "Title": title.encode('utf-8'),
                "Priority": "high",
                "Tags": "chart_with_upwards_trend"
            }
        )
        urllib.request.urlopen(req)
        print(f"📱 Notification sent: {title}")
    except Exception as e:
        print(f"Notification error: {e}")

# OpenAI
import os
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
client = openai.OpenAI(api_key=OPENAI_API_KEY)

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
        
        # Trend
        if len(closes) >= 5:
            trend = "UP" if closes[-1] > closes[-5] else "DOWN"
        else:
            trend = "NEUTRAL"
        
        # RSI
        gains, losses = [], []
        for i in range(1, len(closes)):
            diff = closes[i] - closes[i-1]
            gains.append(diff if diff > 0 else 0)
            losses.append(abs(diff) if diff < 0 else 0)
        
        avg_gain = sum(gains[-14:]) / 14 if len(gains) >= 14 else sum(gains) / len(gains) if gains else 0
        avg_loss = sum(losses[-14:]) / 14 if len(losses) >= 14 else sum(losses) / len(losses) if losses else 0.0001
        rsi = round(100 - (100 / (1 + avg_gain / avg_loss)), 1) if avg_loss > 0 else 50
        
        return {"price": round(price, 5), "trend": trend, "rsi": rsi}
        
    except Exception as e:
        print(f"Error: {e}")
        return None

def analyze(symbol_name: str, price: float, trend: str, rsi: float):
    """تحليل بالذكاء الاصطناعي"""
    
    prompt = f"""Price: {price}, RSI: {rsi}, Trend: {trend}

If opportunity, reply EXACTLY (6 lines):
🟢 BUY
{symbol_name}
Entry: {price}
SL: number
TP: number
Lot: 0.01

Or for SELL:
🔴 SELL
{symbol_name}
Entry: {price}
SL: number
TP: number
Lot: 0.01

If no opportunity: ⏳

NO TEXT. NO ANALYSIS. ONLY 6 LINES OR ⏳"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=50,
        temperature=0.0
    )
    
    return response.choices[0].message.content.strip()

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
async def analyze_endpoint(symbol: str = Query(...), name: str = Query(...)):
    """API endpoint للتحليل"""
    
    # جلب البيانات
    data = get_market_data(symbol)
    
    if not data:
        return {"error": "Failed to fetch data", "price": None, "signal": "⏳"}
    
    # تحليل
    signal_text = analyze(name, data["price"], data["trend"], data["rsi"])
    
    # تحليل الرد
    result = parse_signal(signal_text, data["price"])
    
    # إرسال إشعار إذا في فرصة
    if "BUY" in signal_text or "SELL" in signal_text:
        signal_type = "🟢 شراء" if "BUY" in signal_text else "🔴 بيع"
        msg = f"{name}\nEntry: {result.get('entry', data['price'])}\nSL: {result.get('sl', '--')}\nTP: {result.get('tp', '--')}\nLot: {result.get('lot', '0.01')}"
        send_notification(signal_type, msg)
    
    return result

@app.get("/")
async def root():
    return FileResponse("index.html")

# Static files
app.mount("/", StaticFiles(directory="."), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

