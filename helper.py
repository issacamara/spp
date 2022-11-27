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

        stock_prices = yf.download(symbol, start=start, end=end).reset_index()[['Date', 'Close']]
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
            return df[['Date', 'Ticker', 'Close']]

        results = Parallel(n_jobs=num_cores)(delayed(custom_download)(t, start, end) for t in tickers)
        stock_prices = pd.concat(results, axis=0)

        # stock_prices = yf.download(symbol, start=start, end=end).reset_index()[['Date', 'Close']]
        stock_prices['Date'] = stock_prices['Date'].astype(str)
        res = pd.merge(res, stock_prices, how='inner', left_on=['Date', 'Ticker'], right_on=['Date', 'Ticker'])
        # res['Ticker'] = ticker

        return res

start = date(2022, 11, 16)
end = date(2022, 11, 17)
# print('Done')
data = get_stock_news('TSLA',start, end)

# scrape_google_finance(ticker="GOOGL:NASDAQ")
