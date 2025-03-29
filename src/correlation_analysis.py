import pandas as pd
import os

def load_processed_data():
    data_dir = "data/processed"
    if not os.path.exists(data_dir):
        print(f"[!] Processed data folder '{data_dir}' does not exist!")
        return {}

    data = {}
    for file in os.listdir(data_dir):
        if file.endswith(".csv"):
            try:
                df = pd.read_csv(os.path.join(data_dir, file), parse_dates=["Date"], index_col="Date", 
                                 date_format='%Y-%m-%d')
                symbol = file.replace(".csv", "")
                data[symbol] = df
            except Exception as e:
                print(f"[!] Error loading {file}: {str(e)}")
    return data

def compute_correlation(data):
    # Create a DataFrame for daily returns, including SP500
    return_df = pd.DataFrame({symbol: df["Daily Return"] for symbol, df in data.items()})
    
    # Calculate correlation with S&P 500
    correlation_matrix = return_df.corr()
    
    # Check if SP500 is in the correlation matrix
    if 'SP500' in correlation_matrix:
        sp500_corr = correlation_matrix['SP500']
        return sp500_corr
    else:
        print("[!] SP500 not found in the correlation matrix.")
        return None

def save_correlation_results(correlation_results, filename="correlation_results.csv"):
    if correlation_results is not None:
        correlation_df = pd.DataFrame(correlation_results).reset_index()
        correlation_df.columns = ['Stock', 'Correlation with SP500']
        correlation_df.to_csv(filename, index=False)
        print(f"[+] Saved correlation results to {filename}")
    else:
        print("[!] No correlation results to save.")

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

    # Compute correlation
    correlation_results = compute_correlation(data)

    # Output the correlation results
    print("\nCorrelation with S&P 500:")
    print(correlation_results)

    # Save the results to a CSV file
    save_correlation_results(correlation_results)
