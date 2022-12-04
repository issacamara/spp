#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 11 17:14:25 2022

@author: issacamara
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from datetime import timedelta, date

from sklearn.preprocessing import MinMaxScaler
import config as conf
from tensorflow import keras
import os
from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, Integer, String, DateTime, create_engine, text
import yaml
import yfinance as yf
import multiprocessing
from joblib import Parallel, delayed

Base = declarative_base()


class StockNews(Base):
    __tablename__ = 'stock_news'  # if you use base it is obligatory

    id = Column(Integer, primary_key=True)  # obligatory
    ticker = Column(String)
    exchange = Column(String)
    news = Column(String)
    date = Column(DateTime)

    def __repr__(self):
        return f"<Stock(id='{self.id}', ticker='{self.ticker}', exchange='{self.exchange}, " \
               f"news='{self.news}', date='{self.date}')>"


@dataclass
class Stock():
    ticker: str
    exchange: str
    news: str
    date: str
    price: float


today = date.today()
delta = timedelta(days=2)
symbol = 'GOOG'
offset = conf.offset

path_to_models = conf.models_path

symbols = [s for s in os.listdir(path_to_models) if not s.startswith('.')]

models = {}

for s in symbols:
    models[s] = keras.models.load_model(path_to_models + s)

print("Write this everytime it runs -- Helper")
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
    model = models.get(symbol, models.get('GOOG'))

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


def get_stock_news(ticker, start, end):
    with open('configuration.yml') as f:
        yml = yaml.load(f, Loader=yaml.FullLoader)
        db = yml['database']['name']
        host: str = yml['database']['host']
        user = yml['database']['username']
        pwd = yml['database']['password']
        port = yml['database']['port']
        dialect = yml['database']['dialect']
        driver = yml['database']['driver']
        engine = create_engine(f"{dialect}+{driver}://{user}:{pwd}@{host}:{port}/{db}")

        # session = Session(engine)

        # stocks = session.query(StockNews)\
        #                  .filter(StockNews.date.between(start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d')))\
        #                  .filter(StockNews.ticker == ticker)
        sql = text(f"""select * from spp.stock_news 
                        where ticker='{ticker}' and date between '{start.strftime('%Y-%m-%d')}' 
                            and '{end.strftime('%Y-%m-%d')}' 
                        """)
        # and ticker = '{ticker}'
        stocks = engine.execute(sql)

        data = [(s.ticker, s.exchange, s.news, s.date.strftime('%Y-%m-%d')) for s in stocks]
        df = pd.DataFrame(data, columns=['Ticker', 'Exchange', 'News', 'Date'])
        res = df.groupby(['Ticker', 'Date']).agg({'News': lambda x: ' , '.join(set(x.dropna()))}).reset_index()

        stock_prices = yf.download(symbol, start=start, end=end).reset_index()
        stock_prices['Date'] = stock_prices['Date'].astype(str)
        stock_prices['Ticker'] = ticker
        res = pd.merge(res, stock_prices, how='inner', left_on=['Date', 'Ticker'], right_on=['Date', 'Ticker'])
        # res['Ticker'] = ticker

        return res


def get_all_stock_news(start, end):
    with open('configuration.yml') as f:
        yml = yaml.load(f, Loader=yaml.FullLoader)
        db = yml['database']['name']
        host: str = yml['database']['host']
        user = yml['database']['username']
        pwd = yml['database']['password']
        port = yml['database']['port']
        dialect = yml['database']['dialect']
        driver = yml['database']['driver']
        engine = create_engine(f"{dialect}+{driver}://{user}:{pwd}@{host}:{port}/{db}")

        # session = Session(engine)

        # stocks = session.query(StockNews)\
        #                  .filter(StockNews.date.between(start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d')))\
        #                  .filter(StockNews.ticker == ticker)
        sql = text(f"""select * from spp.stock_news 
                        where date between '{start.strftime('%Y-%m-%d')}' and '{end.strftime('%Y-%m-%d')}' 
                        """)
        # and ticker = '{ticker}'
        stocks = engine.execute(sql)

        data = [(s.ticker, s.exchange, s.news, s.date.strftime('%Y-%m-%d')) for s in stocks]
        df = pd.DataFrame(data, columns=['Ticker', 'Exchange', 'News', 'Date'])
        res = df.groupby(['Ticker', 'Date']).agg({'News': lambda x: ' , '.join(set(x.dropna()))}).reset_index()
        # session.close()
        num_cores = int(multiprocessing.cpu_count())

        tickers = pd.read_csv('tickers.csv', sep=';')['Symbol'].values

        def custom_download(sym, s, e):
            df = yf.download(sym, s, e).reset_index()
            df['Ticker'] = sym
            df['Date'] = df['Date'].astype(str)
            return df

        results = Parallel(n_jobs=num_cores)(delayed(custom_download)(t, start, end) for t in tickers)
        stock_prices = pd.concat(results, axis=0)

        res = pd.merge(res, stock_prices, how='inner', left_on=['Date', 'Ticker'], right_on=['Date', 'Ticker'])
        # res['Ticker'] = ticker

        return res


def trading_indicators(df):
    # SMA
    df['SMA30'] = df['Close'].rolling(30).mean()
    # CMA
    df['CMA30'] = df['Close'].expanding().mean()
    # EWMA
    df['EWMA30'] = df['Close'].ewm(span=30).mean()


    # Fibonacci Retracement
    price_min = df.Close.min()
    price_max = df.Close.max()
    diff = price_max - price_min
    level1 = price_max - 0.236 * diff
    level2 = price_max - 0.382 * diff
    level3 = price_max - 0.618 * diff



def stochastic_oscillator(df):
    # Stochastic Oscillator
    """ A stochastic oscillator is an indicator that compares a specific closing price of an asset to a range of its
    prices over time – showing momentum and trend strength. It uses a scale of 0 to 100. A reading below 20 generally
    represents an oversold market and a reading above 80 an overbought market. However, if a strong trend is present,
    a correction or rally will not necessarily ensue """
    df['14-high'] = df['High'].rolling(14).max()
    df['14-low'] = df['Low'].rolling(14).min()
    df['%K'] = (df['Close'] - df['14-low']) * 100 / (df['14-high'] - df['14-low'])
    df['%D'] = df['%K'].rolling(3).mean()
    df = df.drop(columns=['14-high', '14-low'])
    return df


def macd(row):
    exp1 = row['Close'].ewm(span=12, adjust=False).mean()
    exp2 = row['Close'].ewm(span=26, adjust=False).mean()
    row['MACD'] = exp1 - exp2
    return row['MACD']

def bollinger_bands(row):
    # Bollinger Bands
    row['TP'] = (row['Close'] + row['Low'] + row['High']) / 3
    row['std'] = row['TP'].rolling(20).std(ddof=0)
    row['MA-TP'] = row['TP'].rolling(20).mean()
    row['BOLU'] = row['MA-TP'] + 2 * row['std']
    row['BOLD'] = row['MA-TP'] - 2 * row['std']
    return row[['BOLU', 'BOLD']]

def rsi(row):
    # RSI
    delta = df['Close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.ewm(com=13, adjust=False).mean()
    ema_down = down.ewm(com=13, adjust=False).mean()
    return ema_up / ema_down

def atr(row):
    # Average True Range
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    return true_range.rolling(14).sum() / 14

# import matplotlib.pyplot as plt
# plt.style.use('default')
# start = date(2021, 11, 16)
# end = date(2022, 11, 17)
# df = yf.download('TSLA',start,end)
# df = stochastic_oscillator(df)
# df['SMA30'] = df['Close'].rolling(30).mean()
# df[['Close', 'SMA30']].plot(label='RELIANCE', figsize=(16, 8))

# print('Done')
# data = get_stock_news('TSLA', start, end)

# scrape_google_finance(ticker="GOOGL:NASDAQ")
