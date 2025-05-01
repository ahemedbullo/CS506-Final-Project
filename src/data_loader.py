# src/data_loader.py
import yfinance as yf
import os

def fetch_stock_data(tickers, start_date, end_date, save_path="data/raw/"):
    """Fetch historical stock data from Yahoo Finance."""
    if not os.path.exists(save_path):
        print(f"[*] Creating directory: {save_path}")
        os.makedirs(save_path)
    else:
        print(f"[*] Directory exists: {save_path}")

    all_data = {}
    fetched_tickers = []
    failed_tickers = []

    for ticker in tickers:
        print(f"[*] Fetching data for {ticker}...")
        try:
            # Download data
            stock = yf.download(ticker, start=start_date, end=end_date, progress=False)

            if stock.empty:
                print(f"[!] Warning: No data downloaded for {ticker}. Skipping.")
                failed_tickers.append(ticker)
                continue

            # Save to CSV
            file_path = os.path.join(save_path, f"{ticker}.csv")
            stock.to_csv(file_path)
            all_data[ticker] = stock
            fetched_tickers.append(ticker)
            print(f"[+] Saved data for {ticker} to {file_path}")

        except Exception as e:
            print(f"[!] Error fetching or saving data for {ticker}: {e}")
            failed_tickers.append(ticker)

    print("\n--- Fetch Summary ---")
    print(f"Successfully fetched: {len(fetched_tickers)} tickers")
    print(f"Failed to fetch: {len(failed_tickers)} tickers {failed_tickers if failed_tickers else ''}")
    print("---------------------\n")
    return all_data

if __name__ == "__main__":
    # Expanded list of diverse tickers + S&P 500 index
    tickers = [
        "^GSPC",  # S&P 500 Index (MUST be first or handled explicitly)
        # Tech
        "AAPL", "MSFT", "NVDA", "GOOGL",
        # Healthcare
        "JNJ", "UNH", "LLY", "PFE",
        # Finance
        "JPM", "BAC", "WFC", "BRK-B",
        # Consumer Discretionary
        "AMZN", "HD",
        # Consumer Staples
        "PG", "COST", "WMT", "KO",
        # Energy
        "XOM", "CVX",
        # Industrials
        "CAT", "HON",
        # Utilities
        "NEE",
        # Materials
        "LIN",
        # Real Estate
        "AMT"
        # Add/remove as desired, aim for ~15-25 diverse stocks
    ]
    # Using slightly more recent end date to ensure data availability
    start_date = "2020-01-01"
    end_date = "2024-04-30" # Adjusted end date slightly

    print(f"[*] Starting data download for {len(tickers)} tickers ({start_date} to {end_date})...")
    fetch_stock_data(tickers, start_date, end_date)
    print("[*] Data fetching process complete.")