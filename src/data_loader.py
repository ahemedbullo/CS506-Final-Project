import yfinance as yf
import os

def fetch_stock_data(tickers, start_date, end_date, save_path="data/raw/"):
    """Fetch historical stock data from Yahoo Finance."""
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    all_data = {}
    for ticker in tickers:
        print(f"Fetching data for {ticker}...")
        stock = yf.download(ticker, start=start_date, end=end_date)
        file_path = os.path.join(save_path, f"{ticker}.csv")
        stock.to_csv(file_path)
        all_data[ticker] = stock
    return all_data

if __name__ == "__main__":
    tickers = ["^GSPC", "MSFT", "TSLA", "NVDA"]
    start_date = "2020-01-01"
    end_date = "2024-01-01"
    fetch_stock_data(tickers, start_date, end_date)
