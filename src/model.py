import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from visualization import plot_predicted_vs_actual

def load_processed_data():
    data_dir = "data/processed"
    if not os.path.exists(data_dir):
        print(f"[!] Processed data folder '{data_dir}' does not exist!")
        return {}
    data = {}
    for file in os.listdir(data_dir):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(data_dir, file), parse_dates=["Date"], index_col="Date", date_format='%Y-%m-%d')
            symbol = file.replace(".csv", "")
            data[symbol] = df
    return data

def prepare_features(data, feature_stocks, target_stock="SP500"):
    df = pd.DataFrame({symbol: data[symbol]["Daily Return"] for symbol in feature_stocks + [target_stock]})
    df.dropna(inplace=True)
    X = df[feature_stocks].values
    y = df[target_stock].values
    return X, y, df.index

def weighted_average_baseline(data, weights):
    df = pd.DataFrame({symbol: data[symbol]["Daily Return"] for symbol in weights.keys()})
    df.dropna(inplace=True)
    weighted_avg = np.dot(df.values, np.array(list(weights.values())))
    return weighted_avg, df.index

if __name__ == "__main__":
    print("[*] Loading processed stock data for modeling...")
    data = load_processed_data()
    if "SP500" not in data:
        if "^GSPC" in data:
            data["SP500"] = data["^GSPC"]
            print("[*] Detected ^GSPC.csv and using it as SP500.")
        else:
            print("[!] Missing S&P 500 data (expected SP500.csv or ^GSPC.csv).")
            exit(1)
    feature_stocks = ["MSFT", "TSLA", "NVDA"]
    X, y, idx = prepare_features(data, feature_stocks, target_stock="SP500")
    # 80/20 split, no validation set, shuffle=False to preserve time order
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    print(f"Train/Test split: {X_train.shape[0]} train, {X_test.shape[0]} test")
    # Model: Linear Regression
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # Cross-validation on training set
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    print(f"Cross-validated R^2 (train): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    # Evaluation
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"Test R^2: {r2:.3f}")
    print(f"Test MAE: {mae:.4f}")
    print(f"Test RMSE: {rmse:.4f}")
    # Baseline: Weighted average (static, as per feedback)
    weights = {"MSFT": 0.07, "TSLA": 0.02, "NVDA": 0.05}  # Example weights
    baseline_pred, baseline_idx = weighted_average_baseline(data, weights)
    y_baseline = baseline_pred[-len(y_test):]
    r2_base = r2_score(y_test, y_baseline)
    mae_base = mean_absolute_error(y_test, y_baseline)
    rmse_base = np.sqrt(mean_squared_error(y_test, y_baseline))
    print(f"Baseline (Weighted Avg) Test R^2: {r2_base:.3f}")
    print(f"Baseline (Weighted Avg) Test MAE: {mae_base:.4f}")
    print(f"Baseline (Weighted Avg) Test RMSE: {rmse_base:.4f}")
    # Save results for reporting
    results_df = pd.DataFrame({
        "Date": idx[-len(y_test):],
        "Actual": y_test,
        "Predicted": y_pred,
        "Baseline": y_baseline
    })
    results_df.to_csv("model_results.csv", index=False)
    print("[+] Saved model results to model_results.csv")
    # Visualization
    plot_predicted_vs_actual(y_test, y_pred, title="Predicted vs Actual S&P 500 Returns", filename="predicted_vs_actual.png")
    print("[+] Saved predicted vs actual plot.")
    # Print summary table
    print("\nPerformance Summary:")
    print(f"{'Model':<25}{'R^2':>8}{'MAE':>10}{'RMSE':>10}")
    print(f"{'Linear Regression':<25}{r2:8.3f}{mae:10.4f}{rmse:10.4f}")
    print(f"{'Weighted Avg Baseline':<25}{r2_base:8.3f}{mae_base:10.4f}{rmse_base:10.4f}")
    print("\nDone.")
