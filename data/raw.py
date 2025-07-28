# %%
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yaml
from sklearn.preprocessing import StandardScaler




tickers = {
    "SPY": "S&P 500 ETF",
    "^VIX": "CBOE Volatility Index",
    "QQQ": "Nasdaq 100 ETF"
}






def download_fin_data(tickers,lookback_days):
    end_date= datetime.today()
    start_date=end_date-timedelta(days=lookback_days) #Considering the last 90 days for forecasting
    df=yf.download(list(tickers.keys()),start=start_date,end=end_date)
    df=df["Close"].dropna().reset_index()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date")

    return df
    
 









