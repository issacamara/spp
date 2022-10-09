from sqlalchemy import Column, Integer, String, DateTime, create_engine
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import Session
import datetime as dt
from bs4 import BeautifulSoup
import requests
import pandas as pd
import time
import logging
import multiprocessing
from joblib import Parallel, delayed

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


def scrape_google_finance(ticker: str, exchange: str, date):

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
    news_list = [n.contents[0] for n in news_set]
    news = ''.join(news_list)
    # return StockNews(ticker=ticker, exchange=exchange, news=bytes(news, 'utf8'), date=date)
    return ticker, exchange, news, date


def main():
    tickers = pd.read_csv('tickers.csv', sep=';')
    stocks = tickers[tickers['Market Cap'] > 5000000000][['Symbol', 'Exchange']].values

    date = dt.datetime.now()
    num_cores = multiprocessing.cpu_count()
    results = Parallel(n_jobs=num_cores)(delayed(scrape_google_finance)(stock[0], stock[1], date) for stock in stocks)
    items = [StockNews(ticker=t, exchange=e, news=bytes(n, 'utf8'), date=d) for t, e, n, d in results]


    # sppapp12345@gmail.com spp1234@

    db = 'spp'
    host: str = 'localhost'
    user = 'root'
    pwd = 'root1234'
    port = 3306
    dialect = 'mysql'
    driver = 'pymysql'

    engine = create_engine("{dialect}+{driver}://{user}:{pwd}@{host}:{port}/{db}" \
                           .format(dialect=dialect, driver=driver, user=user, host=host, pwd=pwd, db=db, port=port))
    session = Session(engine)
    session.add_all(items)
    session.commit()
    session.close()
    end = time.time()
    size = len(items)
    logging.info(f'Table successfully loaded with {size} records')
    logging.info(str(dt.timedelta(seconds=(end - start))))


if __name__ == "__main__":
    main()