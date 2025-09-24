import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def load_and_engineer_features(data):
    """Load data and create all engineered features"""
    
    data = data.copy()
    
    try:
        data['VIX_ret'] = data['^VIX'].pct_change()
    except KeyError:
        print("VIX column not found.")

    data['SPY_ret'] = data['SPY'].pct_change()
    data['QQQ_ret'] = data['QQQ'].pct_change()
    data['SPY_vol'] = data['SPY_ret'].rolling(window=5).std()
    data['QQQ_vol'] = data['QQQ_ret'].rolling(window=5).std()
    data['QQQ_SPY_spread'] = (data['QQQ'] - data['SPY']) / data['SPY']
    
    data = data.rename(columns={"^VIX": "VIX"})
    data = data.dropna()
    
    vix_future_change = data['VIX'].pct_change().shift(-1)
    vix_signal = pd.cut(vix_future_change, bins=[-np.inf, -0.01, 0.01, np.inf], labels=[0, 1, 2])
    data['vix_signal'] = vix_signal.astype('Int64')
    
    data = data[:-1]
    
    feature_cols = ['SPY', 'VIX', 'SPY_ret', 'QQQ_ret', 'SPY_vol', 'QQQ_vol', 'QQQ_SPY_spread']
    target_col = 'vix_signal'
    
    return data, feature_cols, target_col