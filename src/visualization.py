# visualizetion.py

import matplotlib.pyplot as plt
import pandas as pd
import os

# Directory to save the plot
VIS_DIR = os.path.join("data", "visualizations")
os.makedirs(VIS_DIR, exist_ok=True)


def load_processed_data():
    data_dir = "data/processed"
    if not os.path.exists(data_dir):
        print(f"[!] Processed data folder '{data_dir}' does not exist!")
        return {}

    data = {}
    for file in os.listdir(data_dir):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(data_dir, file), parse_dates=["Date"])
            symbol = file.replace(".csv", "")
            data[symbol] = df
    return data


def plot_combined_time_series(data, symbols, normalize=False, filename="combined_time_series.png"):
    """
    Plots the closing prices of multiple stocks (including SP500) over time using explicit Date column.

    Args:
        data (dict): Dictionary of DataFrames keyed by stock symbol.
        symbols (list): List of stock symbols to plot.
        normalize (bool): Whether to normalize prices to start at 1.0.
        filename (str): Output file name.
    """
    plt.figure(figsize=(16, 9))

    for symbol in symbols:
        df = data.get(symbol)
        if df is not None and "Close" in df.columns and "Date" in df.columns:
            series = df["Close"]
            if normalize:
                series = series / series.iloc[0]
            plt.plot(df["Date"], series, label=f"{symbol} (start={df['Close'].iloc[0]:.2f})")

    plt.title("Closing Price Trends: S&P 500 vs Key Stocks", fontsize=16)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Normalized Price" if normalize else "Close Price (USD)", fontsize=12)
    plt.legend(title="Stock", fontsize=10, title_fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    save_path = os.path.join(VIS_DIR, filename)
    plt.savefig(save_path)
    print(f"[+] Saved combined time series plot to {save_path}")
    plt.close()


if __name__ == "__main__":
    print("[*] Loading processed stock data...")
    data = load_processed_data()

    if "SP500" not in data:
        if "^GSPC" in data:
            data["SP500"] = data["^GSPC"]
            print("[*] Detected ^GSPC.csv and using it as SP500.")
        else:
            print("[!] Missing S&P 500 data (expected SP500.csv or ^GSPC.csv).")
            exit(1)

    print("[*] Generating combined time series plot...")

    key_symbols = ["SP500", "MSFT", "TSLA", "NVDA"]

    # Plot using Date column (set normalize=True to compare relative change)
    plot_combined_time_series(data, key_symbols, normalize=True)

    print("\nTime series visualization saved in data/visualizations/")
