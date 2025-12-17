import yfinance as yf
import numpy as np

def predict_price(model, scaler, stock="AAPL"):
    data = yf.download(stock, period="3mo")['Close'].values.reshape(-1, 1)
    data = scaler.transform(data)

    last_60 = np.array([data[-60:]])
    prediction = model.predict(last_60)

    return round(float(scaler.inverse_transform(prediction)[0][0]), 2)
