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
import matplotlib.pyplot as plt


stock_list = ['7105.KL']
data = yf.download(stock_list, start="2018-01-01", end="2022-01-01", 
                   interval="1d")
data.head()

training = data.iloc[:,1:2].values

length = training.shape[0]

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,1))

training_scaled = scaler.fit_transform(training)

X_train = []
y_train = []

offset = 30
for i in range(offset,length):
    X_train.append(training_scaled[i-offset:i,0])
    y_train.append(training_scaled[i,0])
    
X_train = np.array(X_train)
y_train = np.array(y_train)
X_train = np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))

from keras.models import Sequential 
from keras.layers import LSTM, Dropout, Dense
regressor = Sequential()
regressor.add(LSTM(units=5, return_sequences=True, 
                   input_shape=(X_train.shape[1],1)))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units=5, return_sequences=True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units=5, return_sequences=True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units=5, return_sequences=True))
regressor.add(Dropout(0.2))
regressor.add(Dense(units=1))

regressor.compile(optimizer='adam', loss='mean_squared_error')
regressor.fit(X_train, y_train, epochs=5, batch_size=32)

