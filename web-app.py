#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 20:35:25 2022

@author: issacamara
"""

import streamlit as st
import pandas as pd
import numpy as np
import datetime
import yfinance as yf


st.title('Stock Price Prediction')
st.markdown("<h1 style='text-align: center; color: red;'>Some title</h1>", 
            unsafe_allow_html=True)

hide_st_style = ("<style>" 
                 # + "#MainMenu {visibility: hidden;}" 
                 +"footer {visibility: hidden;}"
                 # +"header {visibility: hidden;}"
                 +"</style>"
                )


st.markdown(hide_st_style,unsafe_allow_html=True)


sidebar = st.sidebar
sidebar.write('Stock Settings')
option = sidebar.selectbox('Select a stock',  ('GOOG', 'AAPL', 'TSLA'))
sidebar.write('You selected:', option)

col1, col2 = sidebar.columns(2, gap='medium')
d1 = col1.date_input("From",datetime.date(2019, 7, 6))
d2 = col2.date_input("To",datetime.date(2020, 7, 6))
st.write('Your birthday is:', d1)

_, col, _ = sidebar.columns(3, gap='small')

if col.button('Forecast'):
     st.write('Why hello there')
     start = d1.strftime('%Y-%m-%d')
     end = d2.strftime('%Y-%m-%d')
     stock_data = yf.download('GOOG', start=start, end=end)
     chart_data = pd.DataFrame(np.random.randn(50, 3),
                               columns=["a", "b", "c"])

     st.line_chart(stock_data['Close'])
else:
     st.write('Goodbye')


