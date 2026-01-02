from fastapi import FastAPI, UploadFile, File, HTTPException, Header, Depends
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import cv2
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import io
import base64
from typing import Optional, List, Dict
import talib
from pydantic import BaseModel
import json
import warnings
warnings.filterwarnings('ignore')

# API Configuration
API_KEY = "SECRET123"
app = FastAPI(title="AI Trading Application", version="1.0.0")

# Data models
class StockData(BaseModel):
    symbol: str
    start_date: str
    end_date: str

class PredictionRequest(BaseModel):
    symbol: str
    days_ahead: int = 7

class TradeSignal(BaseModel):
    symbol: str
    action: str  # BUY, SELL, HOLD
    confidence: float
    price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

# Dependency for API key authentication
async def verify_api_key(api_key: str = Header(None)):
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return True

# ---- Chart Pattern Scanner ----
def scan_chart_patterns(img):
    """
    Scan trading chart images for technical patterns
    Returns detected patterns and indicators
    """
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Analyze patterns
        patterns = []
        
        for contour in contours:
            if len(contour) > 5:
                # Fit ellipse
                ellipse = cv2.fitEllipse(contour)
                (x, y), (MA, ma), angle = ellipse
                
                # Detect pattern types based on contour properties
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    
                    # Pattern classification
                    if circularity > 0.7:
                        patterns.append({
                            "type": "Head and Shoulders",
                            "confidence": min(0.9, circularity),
                            "center": (int(x), int(y))
                        })
                    elif MA / ma > 2:
                        patterns.append({
                            "type": "Flag/Pennant",
                            "confidence": 0.75,
                            "center": (int(x), int(y))
                        })
        
        # Additional pattern detection using template matching
        patterns_found = {
            "patterns": patterns,
            "edges_detected": int(np.sum(edges > 0)),
            "contours_found": len(contours)
        }
        
        return patterns_found
        
    except Exception as e:
        return {"error": str(e)}

# ---- Technical Analysis Functions ----
def calculate_technical_indicators(df):
    """
    Calculate various technical indicators for trading signals
    """
    # Moving averages
    df['MA20'] = talib.SMA(df['Close'], timeperiod=20)
    df['MA50'] = talib.SMA(df['Close'], timeperiod=50)
    df['MA200'] = talib.SMA(df['Close'], timeperiod=200)
    
    # RSI
    df['RSI'] = talib.RSI(df['Close'], timeperiod=14)
    
    # MACD
    df['MACD'], df['MACD_signal'], df['MACD_hist'] = talib.MACD(
        df['Close'], fastperiod=12, slowperiod=26, signalperiod=9
    )
    
    # Bollinger Bands
    df['BB_upper'], df['BB_middle'], df['BB_lower'] = talib.BBANDS(
        df['Close'], timeperiod=20, nbdevup=2, nbdevdn=2
    )
    
    # Stochastic Oscillator
    df['STOCH_K'], df['STOCH_D'] = talib.STOCH(
        df['High'], df['Low'], df['Close'],
        fastk_period=14, slowk_period=3, slowd_period=3
    )
    
    # ATR for volatility
    df['ATR'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)
    
    return df

def generate_trading_signals(df):
    """
    Generate trading signals based on technical indicators
    """
    signals = []
    
    # Golden Cross / Death Cross
    if len(df) >= 50:
        latest = df.iloc[-1]
        
        # Initialize signal
        signal = {
            "symbol": "N/A",
            "action": "HOLD",
            "confidence": 0.5,
            "price": float(latest['Close']),
            "timestamp": datetime.now().isoformat()
        }
        
        # Moving Average Crossover
        if latest['MA20'] > latest['MA50'] and df.iloc[-2]['MA20'] <= df.iloc[-2]['MA50']:
            signal["action"] = "BUY"
            signal["confidence"] = 0.7
            signal["reason"] = "Golden Cross (20 crossed above 50)"
        elif latest['MA20'] < latest['MA50'] and df.iloc[-2]['MA20'] >= df.iloc[-2]['MA50']:
            signal["action"] = "SELL"
            signal["confidence"] = 0.7
            signal["reason"] = "Death Cross (20 crossed below 50)"
        
        # RSI signals
        if latest['RSI'] < 30:
            signal["action"] = "BUY"
            signal["confidence"] = max(signal["confidence"], 0.65)
            signal["reason"] = "RSI oversold"
        elif latest['RSI'] > 70:
            signal["action"] = "SELL"
            signal["confidence"] = max(signal["confidence"], 0.65)
            signal["reason"] = "RSI overbought"
        
        # MACD signals
        if latest['MACD'] > latest['MACD_signal'] and df.iloc[-2]['MACD'] <= df.iloc[-2]['MACD_signal']:
            signal["action"] = "BUY"
            signal["confidence"] = max(signal["confidence"], 0.6)
            signal["reason"] = "MACD bullish crossover"
        elif latest['MACD'] < latest['MACD_signal'] and df.iloc[-2]['MACD'] >= df.iloc[-2]['MACD_signal']:
            signal["action"] = "SELL"
            signal["confidence"] = max(signal["confidence"], 0.6)
            signal["reason"] = "MACD bearish crossover"
        
        # Bollinger Bands
        if latest['Close'] < latest['BB_lower']:
            signal["action"] = "BUY"
            signal["confidence"] = max(signal["confidence"], 0.6)
            signal["reason"] = "Price below lower Bollinger Band"
        elif latest['Close'] > latest['BB_upper']:
            signal["action"] = "SELL"
            signal["confidence"] = max(signal["confidence"], 0.6)
            signal["reason"] = "Price above upper Bollinger Band"
        
        # Add stop loss and take profit levels
        if signal["action"] != "HOLD":
            atr = latest['ATR']
            if signal["action"] == "BUY":
                signal["stop_loss"] = float(latest['Close'] - 2 * atr)
                signal["take_profit"] = float(latest['Close'] + 3 * atr)
            else:
                signal["stop_loss"] = float(latest['Close'] + 2 * atr)
                signal["take_profit"] = float(latest['Close'] - 3 * atr)
        
        signals.append(signal)
    
    return signals

# ---- Prediction Model (Simple) ----
def predict_price_trend(df, days_ahead=7):
    """
    Simple linear regression for price prediction
    """
    from sklearn.linear_model import LinearRegression
    
    # Prepare data
    X = np.arange(len(df)).reshape(-1, 1)
    y = df['Close'].values
    
    # Train model
    model = LinearRegression()
    model.fit(X, y)
    
    # Predict future
    future_days = np.arange(len(df), len(df) + days_ahead).reshape(-1, 1)
    predictions = model.predict(future_days)
    
    # Calculate trend
    current_price = df['Close'].iloc[-1]
    predicted_price = predictions[-1]
    trend = "BULLISH" if predicted_price > current_price else "BEARISH"
    
    return {
        "current_price": float(current_price),
        "predicted_prices": [float(p) for p in predictions],
        "trend": trend,
        "confidence": min(0.85, abs(predicted_price - current_price) / current_price * 10)
    }

# ---- API Endpoints ----
@app.post("/api/scan-chart")
async def scan_chart_image(
    file: UploadFile = File(...),
    authenticated: bool = Depends(verify_api_key)
):
    """
    Upload and analyze trading chart images
    """
    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Scan for patterns
        results = scan_chart_patterns(img)
        
        return JSONResponse(content={
            "status": "success",
            "filename": file.filename,
            "patterns_detected": results.get("patterns", []),
            "analysis": {
                "edge_pixels": results.get("edges_detected", 0),
                "contours": results.get("contours_found", 0)
            }
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analyze-stock")
async def analyze_stock(
    data: StockData,
    authenticated: bool = Depends(verify_api_key)
):
    """
    Analyze stock data with technical indicators
    """
    try:
        # Download stock data
        ticker = yf.Ticker(data.symbol)
        df = ticker.history(start=data.start_date, end=data.end_date)
        
        if df.empty:
            raise HTTPException(status_code=404, detail="No data found for symbol")
        
        # Calculate technical indicators
        df = calculate_technical_indicators(df)
        
        # Generate trading signals
        signals = generate_trading_signals(df)
        
        # Calculate basic statistics
        latest = df.iloc[-1]
        stats = {
            "current_price": float(latest['Close']),
            "daily_change": float(latest['Close'] - df.iloc[-2]['Close']),
            "daily_change_pct": float((latest['Close'] - df.iloc[-2]['Close']) / df.iloc[-2]['Close'] * 100),
            "volume": int(latest['Volume']),
            "avg_volume": int(df['Volume'].tail(20).mean()),
            "volatility": float(df['Close'].tail(20).std())
        }
        
        return JSONResponse(content={
            "status": "success",
            "symbol": data.symbol,
            "analysis_date": datetime.now().isoformat(),
            "signals": signals,
            "statistics": stats,
            "indicators": {
                "RSI": float(latest['RSI']),
                "MACD": float(latest['MACD']),
                "MA20": float(latest['MA20']),
                "MA50": float(latest['MA50']),
                "BB_upper": float(latest['BB_upper']),
                "BB_lower": float(latest['BB_lower'])
            }
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/predict")
async def predict_price(
    request: PredictionRequest,
    authenticated: bool = Depends(verify_api_key)
):
    """
    Predict future price trends
    """
    try:
        # Get historical data (last 90 days)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)
        
        ticker = yf.Ticker(request.symbol)
        df = ticker.history(start=start_date.strftime('%Y-%m-%d'), 
                           end=end_date.strftime('%Y-%m-%d'))
        
        if df.empty:
            raise HTTPException(status_code=404, detail="No data found for symbol")
        
        # Make prediction
        prediction = predict_price_trend(df, request.days_ahead)
        
        return JSONResponse(content={
            "status": "success",
            "symbol": request.symbol,
            "prediction": prediction,
            "prediction_horizon": f"{request.days_ahead} days",
            "timestamp": datetime.now().isoformat()
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/market-scan")
async def market_scan(
    authenticated: bool = Depends(verify_api_key)
):
    """
    Scan top stocks for trading opportunities
    """
    try:
        # List of popular stocks to scan
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX']
        
        results = []
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(period="1mo")
                
                if not df.empty:
                    df = calculate_technical_indicators(df)
                    signals = generate_trading_signals(df)
                    
                    if signals and signals[0]["action"] != "HOLD":
                        results.append({
                            "symbol": symbol,
                            "signal": signals[0]["action"],
                            "confidence": signals[0]["confidence"],
                            "price": float(df['Close'].iloc[-1]),
                            "change": float(df['Close'].iloc[-1] - df['Close'].iloc[-2])
                        })
            except:
                continue
        
        # Sort by confidence
        results.sort(key=lambda x: x["confidence"], reverse=True)
        
        return JSONResponse(content={
            "status": "success",
            "scan_time": datetime.now().isoformat(),
            "stocks_analyzed": len(symbols),
            "opportunities_found": len(results),
            "top_opportunities": results[:5]  # Return top 5
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/portfolio-analysis")
async def portfolio_analysis(
    symbols: str,  # Comma-separated list of symbols
    authenticated: bool = Depends(verify_api_key)
):
    """
    Analyze multiple stocks as a portfolio
    """
    try:
        symbol_list = [s.strip().upper() for s in symbols.split(',')]
        
        portfolio_data = []
        total_value = 0
        
        for symbol in symbol_list:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period="1mo")
            
            if not df.empty:
                current_price = float(df['Close'].iloc[-1])
                df = calculate_technical_indicators(df)
                signals = generate_trading_signals(df)
                
                stock_analysis = {
                    "symbol": symbol,
                    "current_price": current_price,
                    "signal": signals[0]["action"] if signals else "HOLD",
                    "rsi": float(df['RSI'].iloc[-1]),
                    "volatility": float(df['Close'].tail(20).std())
                }
                
                portfolio_data.append(stock_analysis)
                total_value += current_price
        
        # Calculate portfolio metrics
        if portfolio_data:
            avg_rsi = np.mean([s["rsi"] for s in portfolio_data])
            bull_count = len([s for s in portfolio_data if s["signal"] == "BUY"])
            bear_count = len([s for s in portfolio_data if s["signal"] == "SELL"])
            
            return JSONResponse(content={
                "status": "success",
                "portfolio_size": len(portfolio_data),
                "total_estimated_value": total_value,
                "market_sentiment": "BULLISH" if bull_count > bear_count else "BEARISH",
                "average_rsi": float(avg_rsi),
                "bullish_stocks": bull_count,
                "bearish_stocks": bear_count,
                "neutral_stocks": len(portfolio_data) - bull_count - bear_count,
                "detailed_analysis": portfolio_data
            })
        else:
            raise HTTPException(status_code=404, detail="No valid stock data found")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    """
    API documentation and usage instructions
    """
    html_content = """
    <html>
        <head>
            <title>AI Trading Application</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .endpoint { background: #f4f4f4; padding: 10px; margin: 10px 0; }
                code { background: #e0e0e0; padding: 2px 4px; }
            </style>
        </head>
        <body>
            <h1>AI Trading Application API</h1>
            <p>This API provides AI-powered trading analysis tools.</p>
            
            <h2>Available Endpoints:</h2>
            
            <div class="endpoint">
                <h3>POST /api/scan-chart</h3>
                <p>Upload trading chart images for pattern recognition</p>
                <p><strong>Headers:</strong> api-key: SECRET123</p>
            </div>
            
            <div class="endpoint">
                <h3>POST /api/analyze-stock</h3>
                <p>Technical analysis of stock symbols</p>
                <p><strong>Body:</strong> { "symbol": "AAPL", "start_date": "2024-01-01", "end_date": "2024-12-31" }</p>
            </div>
            
            <div class="endpoint">
                <h3>POST /api/predict</h3>
                <p>Price prediction for given symbols</p>
                <p><strong>Body:</strong> { "symbol": "AAPL", "days_ahead": 7 }</p>
            </div>
            
            <div class="endpoint">
                <h3>GET /api/market-scan</h3>
                <p>Scan top stocks for trading opportunities</p>
            </div>
            
            <div class="endpoint">
                <h3>GET /api/portfolio-analysis</h3>
                <p>Analyze multiple stocks as portfolio</p>
                <p><strong>Query Param:</strong> symbols=AAPL,GOOGL,MSFT</p>
            </div>
            
            <h2>Usage Example:</h2>
            <code>
            curl -X POST "http://localhost:8000/api/analyze-stock" \
                 -H "api-key: SECRET123" \
                 -H "Content-Type: application/json" \
                 -d '{"symbol": "AAPL", "start_date": "2024-01-01", "end_date": "2024-12-31"}'
            </code>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content)

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
