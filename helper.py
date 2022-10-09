#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 11 17:14:25 2022

@author: issacamara
"""

import numpy as np
import pandas as pd
from datetime import  timedelta, date
from sklearn.preprocessing import MinMaxScaler
import config as conf
from tensorflow import keras
import os



today = date.today()
delta = timedelta(days=2)
symbol = 'GOOG'
offset = conf.offset

path_to_models = conf.models_path

symbols = [s for s in os.listdir(path_to_models) if not s.startswith('.')]

models = {}

for s in symbols:
    models[s] = keras.models.load_model(path_to_models + s)


# Computes predictors from data up to i-th item, i must be greater than offset
def get_predictors(data, i):
    prev = data[i - offset:i, 0]
    position = 1
    return np.insert(prev, position, np.mean(prev))


def forecast(symbol, stock_data, nb_days):
    # yesterday = date.today() - timedelta(days=1)
    # end = yesterday.strftime('%Y-%m-%d')
    # start= (yesterday - timedelta(days=30)).strftime('%Y-%m-%d')
    # stock_data = yf.download(symbol, start=start, end=end)
    model = models[symbol]

    scaler = MinMaxScaler(feature_range=(0, 1))

    values = stock_data['Close'].values
    scaled_data = scaler.fit_transform(values.reshape(-1, 1))

    new_data = scaled_data.copy()
    df = pd.DataFrame()

    offset = conf.offset

    last_day = max(stock_data.index)
    for i in range(nb_days):
        l = new_data.shape[0]
        x_forecast = np.array(get_predictors(new_data, l))
        # x_forecast = np.array([new_data[l - offset:l, 0]])
        x_forecast = np.reshape(x_forecast, (1, x_forecast.shape[0], 1))
        forecast = model.predict(x_forecast)
        new_data = np.append(new_data, forecast[0]).reshape(-1, 1)
        forecast = scaler.inverse_transform(forecast)
        # print(forecast)
        last_day = last_day + pd.Timedelta(1, unit="d")
        df.loc[last_day, 'Close'] = forecast[0, 0]

    df.index.names = ['Date']
    return df



# scrape_google_finance(ticker="GOOGL:NASDAQ")
