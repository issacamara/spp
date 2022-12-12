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
import cufflinks as cf
import os
print('Value for USER is', st.secrets["db_hostname"])
import helper
from dateutil.relativedelta import relativedelta  # to add days or years


# model = keras.models.load_model(conf.home_path + conf.models_path)

@st.cache
def get_info(t):
    dico = yf.Ticker(t).info
    return {'shortName': dico['shortName'], 'logo_url': dico['logo_url']}


@st.cache
def get_all_tickers():
    return list(pd.read_csv('tickers.csv', sep=';')['Symbol'])


@st.cache
def get_stock_prices(t, s, e):
    return yf.download(t, start=s, end=e)


def center_content(container, color, obj):
    container.markdown(f"<h1 style='text-align: center; color: {color};'>{obj}</h1>", unsafe_allow_html=True)


print("Write this everytime it runs -- WebApp", dt.datetime.now())

title = 'Stock Price Prediction'
# centerize(st,'black', title)
# st.title('Stock Price Prediction')

hide_st_style = ("<style>"
                 # + "#MainMenu {visibility: hidden;}" 
                 + "footer {visibility: hidden;}"
                 # +"header {visibility: hidden;}"
                 + "</style>"
                 )

st.markdown(hide_st_style, unsafe_allow_html=True)

tickers = get_all_tickers()

# temporality = sidebar.selectbox('Select the temporality', ('Minute', 'Day'))
# sidebar.write('You selected:', stock)

############################################### SIDEBAR ###############################################
sidebar = st.sidebar
center_content(sidebar, 'black', 'Stock Settings')
# sidebar.write('Stock Settings')
ticker = sidebar.selectbox('Select a stock', tickers)

from_col, to_col = sidebar.columns(2, gap='medium')
d1 = from_col.date_input("From", dt.date(2016, 7, 6))
d2 = to_col.date_input("To", dt.date(2020, 7, 6))
investment = sidebar.number_input(label='Investment amount (€)', min_value=100, step=10)
format1 = 'DD-MM-YYYY'
format2 = '%Y-%m-%d'
start_date = dt.datetime.now().date() - relativedelta(days=60)  # I need some range in the past
end_date = dt.datetime.now().date() + relativedelta(days=60)
max_days = end_date - start_date
slider = sidebar.slider('Select a range for forecasting', min_value=start_date, value=end_date,
                        max_value=end_date, format=format1)
info = get_info(ticker)
_, col, _ = sidebar.columns(3, gap='small')
logo = f"""<img src={info['logo_url']} alt='{ticker}' width='50' height='50'> """

center_content(sidebar, 'black', logo)
# sidebar.markdown(logo, unsafe_allow_html=True)

############################################### MAIN PANEL ###############################################
center_content(st, 'red', f"""{info['shortName']}""")
start = start_date.strftime(format2)
end = end_date.strftime(format2)
stock_data = get_stock_prices(ticker, start, end)
qf = cf.QuantFig(stock_data, title="Quant Figure", legend='top', name='GS', up_color='green', down_color='red')
    # qf.add_volume(name='Volume',up_color='green', down_color='red')
    # qf.add_adx(periods=20)

history_chart, forecast_chart = st.columns([5, 0.5], gap='large')

ind_container = forecast_chart.container()
ind_container.title("Indicators")
sma = ind_container.checkbox('SMA')
if(sma):
    qf.add_sma(periods=20, legendgroup=True)

ema = ind_container.checkbox('EMA')
if(ema):
    qf.add_ema(periods=20, name='EMA', color='pink', legendgroup=True)

macd = ind_container.checkbox('MACD')
if(macd):
    qf.add_macd(fast_period=12, slow_period=26, signal_period=9, column=None, name='MACD',  color=['blue'])

rsi = ind_container.checkbox('RSI')
if(rsi):
    qf.add_rsi(periods=20, color='java', name='RSI')

adx = ind_container.checkbox('ADX')
if(adx):
    qf.add_adx(periods=20, name='ADX')

atr = ind_container.checkbox('ATR')
if(atr):
    qf. add_atr(periods=20, name='ATR')
bollinger_bands = ind_container.checkbox('Bollinger Bands')
if(bollinger_bands):
    qf.add_bollinger_bands(periods=20, boll_std=2, colors=['magenta', 'grey'], fill=True)

fig = qf.iplot(asFigure=True)
history_chart.plotly_chart(fig, use_container_width=True)

if col.button('Forecast'):
    # st.write('Stock Prices between', start, 'and', end)

    # st.line_chart(stock_data['Close'])
    predictions = helper.forecast(ticker, stock_data, 60)
    df = pd.merge(left=stock_data, right=predictions, how='outer',
                  left_on='Date', right_on='Date')
    df = df.rename(columns={"Close_x": "Actual", "Close_y": "Prediction"})
    # history_chart.line_chart(df[['Actual']])
    # forecast_chart.line_chart(df[['Prediction']])
