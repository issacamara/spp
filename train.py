#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 16:59:13 2022

@author: issacamara
"""

import os
import pandas as pd
import numpy as np
import yfinance as yf
# import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import math
from tensorflow import keras
from tensorflow.keras.layers import LSTM, Dropout, Dense
import config as conf
import helper as h

start_date = '2022-01-01'
end_date = '2022-08-01'
symbol = 'GOOG'
offset = conf.offset
batch_size = 32
epochs = 4
scaler = MinMaxScaler(feature_range=(0, 1))


def train_test_split(data, train_size):
    training_data_len = math.ceil(data.shape[0] * train_size)

    train_data = data[0: training_data_len, :]

    x_train = []
    y_train = []

    for i in range(offset, training_data_len):
        x_train.append(h.get_predictors(train_data,i))
        y_train.append(train_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    x_test = []
    y_test = data[training_data_len:]
    for i in range(training_data_len, len(data)):
        x_test.append(data[i - offset:i, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    return x_train, y_train, x_test, y_test


def train(symbol, start, end, offset, batch_size, epochs, scaler):
    stock_data = yf.download(symbol, start, end)
    values = stock_data['Close'].values

    scaled_data = scaler.fit_transform(values.reshape(-1, 1))

    x_train, y_train, _, _ = train_test_split(scaled_data, 0.8)

    model = keras.Sequential()
    model.add(LSTM(100, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(100, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(25))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)

    model.save(conf.models_path + symbol)


train(symbol, start_date, end_date, offset, batch_size, epochs, scaler)
