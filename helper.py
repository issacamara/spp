#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 11 17:14:25 2022

@author: issacamara
"""
import math
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler 
import matplotlib.pyplot as plt
import tensorflow as tf
from datetime import datetime, timedelta, date
import time
from sklearn.preprocessing import MinMaxScaler 
import config as conf
from tensorflow import keras
import os


today = date.today()
delta = timedelta(days=2)
symbol = 'GOOG'

path_to_models = conf.home_path+conf.models_path

symbols = [s for s in os.listdir(path_to_models) if s != '.DS_Store']

models = {}

for s in symbols:
    models[s] = keras.models.load_model(path_to_models + s)

def forecast(symbol, stock_data, nb_days):
    # yesterday = date.today() - timedelta(days=1)
    # end = yesterday.strftime('%Y-%m-%d')
    # start= (yesterday - timedelta(days=30)).strftime('%Y-%m-%d')
    # stock_data = yf.download(symbol, start=start, end=end)
    model = models[symbol]
    
    scaler = MinMaxScaler(feature_range=(0,1))
    
    values = stock_data['Close'].values
    scaled_data = scaler.fit_transform(values.reshape(-1,1))
    
    new_data = scaled_data.copy()
    df = pd.DataFrame()
    
    offset = conf.offset
    
    last_day = max(stock_data.index)
    for i in range(nb_days):
        l = new_data.shape[0]
        x_forecast = np.array([new_data[l-offset:l, 0]])
        x_forecast = np.reshape(x_forecast, (x_forecast.shape[0], x_forecast.shape[1], 1))
        forecast = model.predict(x_forecast)
        new_data = np.append(new_data,forecast[0]).reshape(-1,1)
        forecast = scaler.inverse_transform(forecast)
        #print(forecast)
        last_day = last_day + pd.Timedelta(1, unit="d")
        df.loc[last_day,'Close'] = forecast[0,0]
        
    
    df.index.names = ['Date']
    return df
