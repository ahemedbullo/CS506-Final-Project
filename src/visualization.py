# visualizetion.py

import matplotlib.pyplot as plt
import seaborn as sns  # type: ignore
import pandas as pd
import os

# Create the data/visualizations directory if it doesn't exist
VIS_DIR = os.path.join("data", "visualizations")
os.makedirs(VIS_DIR, exist_ok=True)


def plot_time_series(df, columns, title="Time Series Plot", ylabel="Price", xlabel="Date", filename="time_series.png"):
    """
    Plots time series of specified columns and saves it as a PNG in data/visualizations/.
    """
    plt.figure(figsize=(14, 7))
    for col in columns:
        plt.plot(df.index, df[col], label=col)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(VIS_DIR, filename)
    plt.savefig(save_path)
    print(f"[+] Saved time series plot to {save_path}")
    plt.close()


def plot_correlation_heatmap(df, title="Correlation Heatmap", filename="correlation_heatmap.png"):
    """
    Plots and saves a heatmap of correlations between DataFrame columns.
    """
    plt.figure(figsize=(12, 8))
    corr = df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title(title)
    plt.tight_layout()

    save_path = os.path.join(VIS_DIR, filename)
    plt.savefig(save_path)
    print(f"[+] Saved correlation heatmap to {save_path}")
    plt.close()


def plot_scatter_with_regression(x, y, x_label, y_label, title="Scatter Plot with Regression", filename="scatter_regression.png"):
    """
    Creates a scatter plot with regression line and saves it.
    """
    plt.figure(figsize=(10, 6))
    sns.regplot(x=x, y=y, scatter_kws={"alpha": 0.5}, line_kws={"color": "red"})
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.tight_layout()

    save_path = os.path.join(VIS_DIR, filename)
    plt.savefig(save_path)
    print(f"[+] Saved scatter regression plot to {save_path}")
    plt.close()


def plot_predicted_vs_actual(y_true, y_pred, title="Predicted vs Actual S&P 500 Returns", filename="predicted_vs_actual.png"):
    """
    Plots predicted vs actual values and saves the result.
    """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.6)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='red', linestyle='--')
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(title)
    plt.tight_layout()

    save_path = os.path.join(VIS_DIR, filename)
    plt.savefig(save_path)
    print(f"[+] Saved predicted vs actual plot to {save_path}")
    plt.close()


def load_processed_data():
    data_dir = "data/processed"
    if not os.path.exists(data_dir):
        print(f"[!] Processed data folder '{data_dir}' does not exist!")
        return {}

    data = {}
    for file in os.listdir(data_dir):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(data_dir, file), parse_dates=["Date"], index_col="Date")
            symbol = file.replace(".csv", "")
            data[symbol] = df
    return data


if __name__ == "__main__":
    print("[*] Loading processed stock data...")
    data = load_processed_data()

    # Handle ^GSPC.csv as SP500 if needed
    if "SP500" not in data:
        if "^GSPC" in data:
            data["SP500"] = data["^GSPC"]
            print("[*] Detected ^GSPC.csv and using it as SP500.")
        else:
            print("[!] Missing S&P 500 data (expected SP500.csv or ^GSPC.csv).")
            exit(1)

    sp500 = data["SP500"]
    sp500_returns = sp500["Daily Return"]

    print("[*] Generating visualizations...")

    # Time series plots
    for symbol, df in data.items():
        plot_time_series(df, ["Close"], f"{symbol} Close Price Over Time", f"{symbol}_time_series.png")

    # Correlation heatmap of daily returns
    return_df = pd.DataFrame({k: v["Daily Return"] for k, v in data.items()})
    plot_correlation_heatmap(return_df, "Daily Returns Correlation", "daily_returns_correlation.png")

    # Regression plots vs S&P 500
    for symbol, df in data.items():
        if symbol == "SP500":
            continue
        plot_scatter_with_regression(
            df["Daily Return"],
            sp500_returns,
            x_label=f"{symbol} Daily Return",
            y_label="S&P 500 Daily Return",
            title=f"{symbol} vs S&P 500",
            filename=f"{symbol}_vs_SP500_regression.png"
        )

    print("\n✅ All visualizations saved in data/visualizations/")
