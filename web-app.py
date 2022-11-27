#!/usr/bin/env python3
# -*- coding: utf-8 -*-∑
"""
Created on Tue Sep  6 20:35:25 2022

@author: issacamara
"""

import streamlit as st
import pandas as pd
import datetime as dt
import yfinance as yf
import helper
from dateutil.relativedelta import relativedelta  # to add days or years

# model = keras.models.load_model(conf.home_path + conf.models_path)

title = 'Stock Price Prediction'
st.markdown(f"<h1 style='text-align: center; color: black;'>{title}</h1>", unsafe_allow_html=True)
# st.title('Stock Price Prediction')

hide_st_style = ("<style>"
                 # + "#MainMenu {visibility: hidden;}" 
                 + "footer {visibility: hidden;}"
                 # +"header {visibility: hidden;}"
                 + "</style>"
                 )

st.markdown(hide_st_style, unsafe_allow_html=True)

sidebar = st.sidebar
sidebar.write('Stock Settings')
tickers = list(pd.read_csv('tickers.csv', sep=';')['Symbol'])

stock = sidebar.selectbox('Select a stock', tickers)
# temporality = sidebar.selectbox('Select the temporality', ('Minute', 'Day'))
# sidebar.write('You selected:', stock)
st.markdown(f"<h1 style='text-align: center; color: red;'>{stock} Stock Prices </h1>",
            unsafe_allow_html=True)
############################################### TABS ###############################################
title1 = "History"
title2 = "Forecast"
# history_tab, forecast_tab = st.tabs([title1, title2])
#
# with history_tab:
#    st.header(title1)
#
# with forecast_tab:
#    st.header(title2)

history_chart, forecast_chart = st.columns(2, gap='medium')
############################################### SIDEBAR ###############################################

col1, col2 = sidebar.columns(2, gap='medium')
d1 = col1.date_input("From", dt.date(2016, 7, 6))
d2 = col2.date_input("To", dt.date(2020, 7, 6))
investment = sidebar.number_input(label='Investment amount (€)', min_value=100, step=10)
format1 = 'DD-MM-YYYY'
format2 = '%Y-%m-%d'
start_date = dt.datetime.now().date() - relativedelta(days=60)  # I need some range in the past
end_date = dt.datetime.now().date() + relativedelta(days=60)
max_days = end_date - start_date
slider = sidebar.slider('Select a range', min_value=start_date, value=end_date, max_value=end_date, format=format1)

_, col, _ = sidebar.columns(3, gap='small')


# st.write(investment)
st.markdown(f"<h1 style='text-align: center; color: blue;'>{investment} </h1>",
            unsafe_allow_html=True)
if col.button('Forecast'):
    start = start_date.strftime(format2)
    end = end_date.strftime(format2)
    stock_data = yf.download(stock, start=start, end=end)

    # st.write('Stock Prices between', start, 'and', end)

    # st.line_chart(stock_data['Close'])
    predictions = helper.forecast(stock, stock_data, 60)
    df = pd.merge(left=stock_data, right=predictions, how='outer',
                  left_on='Date', right_on='Date')
    df = df.rename(columns={"Close_x": "Actual", "Close_y": "Prediction"})
    history_chart.line_chart(df[['Actual']])
    forecast_chart.line_chart(df[['Prediction']])


