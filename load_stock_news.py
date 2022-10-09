from typing import Any

from sqlalchemy import Column, Integer, String, DateTime, create_engine
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import Session
import datetime as dt
from bs4 import BeautifulSoup
import requests
import pandas as pd
import time
import logging

start = time.time()
Base = declarative_base()
# logger = logging.getLogger('Stock News Logger')
logging.basicConfig(level=logging.INFO)


class StockNews(Base):
    __tablename__ = 'stock_news'  # if you use base it is obligatory

    id = Column(Integer, primary_key=True)  # obligatory
    ticker = Column(String)
    exchange = Column(String)
    news = Column(String)
    date = Column(DateTime)


def scrape_google_finance(ticker: str, exchange: str):
    params = {
        "hl": "en"  # language
    }

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.60 Safari/537.36",
    }

    page = requests.get(f"https://www.google.com/finance/quote/{ticker}:{exchange}", params=params, headers=headers,
                        timeout=30)
    soup = BeautifulSoup(page.content, 'html.parser')
    news_set = soup.find_all("div", {"class": "Yfwt5"})
    news = [n.contents[0] for n in news_set]
    return news


def main():
    tickers = pd.read_csv('tickers.csv', sep=';')
    stocks = tickers[tickers['Market Cap'] > 5000000000][['Symbol', 'Exchange']].head().values
    items = []
    date = dt.datetime.now()
    for stock in stocks:
        ticker = stock[0]
        exchange = stock[1]
        # logger.info(ticker, exchange, 'Downloading news ...')
        news = ''.join(scrape_google_finance(ticker, exchange))
        stock = StockNews(ticker=ticker, exchange=exchange, news=bytes(news, 'utf8'), date=date)
        time.sleep(0.05)
        items.append(stock)

    # sppapp12345@gmail.com spp1234@
    db = 'spp'
    host: str = 'localhost'
    pwd = 'root1234'
    user = 'root'
    port = 3306
    dialect = 'mysql'
    driver = 'pymysql'

    engine = create_engine("{dialect}+{driver}://{user}:{pwd}@{host}:{port}/{db}" \
                           .format(dialect=dialect, driver=driver, user=user, host=host, pwd=pwd, db=db, port=port))
    session = Session(engine)
    session.add_all(items)
    # session.commit()
    session.close()
    end = time.time()
    logging.info('Table successfully loaded !')
    logging.info(str(dt.timedelta(seconds=(end - start))))


if __name__ == "__main__":
    main()
