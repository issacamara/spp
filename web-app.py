#!/usr/bin/env python3
# -*- coding: utf-8 -*-∑
"""
Created on Tue Sep  6 20:35:25 2022

@author: issacamara
"""

import streamlit as st
import pandas as pd
import numpy as np
import datetime
import yfinance as yf
import config as conf
from tensorflow import keras
import helper

# model = keras.models.load_model(conf.home_path + conf.models_path)

st.title('Stock Price Prediction')

hide_st_style = ("<style>"
                 # + "#MainMenu {visibility: hidden;}" 
                 + "footer {visibility: hidden;}"
                 # +"header {visibility: hidden;}"
                 + "</style>"
                 )

st.markdown(hide_st_style, unsafe_allow_html=True)



sidebar = st.sidebar
sidebar.write('Stock Settings')
stock = sidebar.selectbox('Select a stock', ('GOOG', 'AAPL', 'TSLA'))
temporality = sidebar.selectbox('Select the temporality', ('MIN', 'DAY'))
# sidebar.write('You selected:', stock)
st.markdown("<h1 style='text-align: center; color: red;'>{} Stock Prices </h1>".format(stock),
            unsafe_allow_html=True)

col1, col2 = sidebar.columns(2, gap='medium')
d1 = col1.date_input("From", datetime.date(2016, 7, 6))
d2 = col2.date_input("To", datetime.date(2020, 7, 6))

_, col, _ = sidebar.columns(3, gap='small')

if col.button('Forecast'):
    start = d1.strftime('%Y-%m-%d')
    end = d2.strftime('%Y-%m-%d')
    stock_data = yf.download(stock, start=start, end=end)

    st.write('Stock Prices between', d1, 'and', d2)

    # st.line_chart(stock_data['Close'])
    predictions = helper.forecast(stock, stock_data, 60)
    df = pd.merge(left=stock_data, right=predictions, how='outer',
                  left_on='Date', right_on='Date')
    df = df.rename(columns={"Close_x": "Actual", "Close_y": "Prediction"})
    st.line_chart(df[['Actual', 'Prediction']])
else:
    st.write('Goodbye')
