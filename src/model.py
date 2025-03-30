# model.py

import pandas as pd
import os
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def load_processed_data():
    processed_dir = "data/processed"
    if not os.path.exists(processed_dir):
        print(f"[!] Processed directory {processed_dir} not found.")
        return {}

    data = {}
    for file in os.listdir(processed_dir):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(processed_dir, file), parse_dates=["Date"], index_col="Date")
            symbol = file.replace(".csv", "")
            data[symbol] = df
    return data


def get_top_correlated_stocks(return_df, target_col="SP500", top_n=5):
    correlations = return_df.corr()[target_col].drop(target_col)
    top_stocks = correlations.abs().sort_values(ascending=False).head(top_n).index.tolist()
    print(f"\n📊 Top {top_n} stocks most correlated with {target_col}:\n")
    print(correlations[top_stocks])
    return top_stocks


def build_and_evaluate_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\n🔍 Model Evaluation:")
    print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.6f}")
    print(f"R² Score: {r2_score(y_test, y_pred):.4f}")

    return model, X.columns


if __name__ == "__main__":
    print("[*] Loading processed stock data...")
    data = load_processed_data()

    # Handle ^GSPC.csv as SP500 if needed
    if "SP500" not in data:
        if "^GSPC" in data:
            data["SP500"] = data["^GSPC"]
            print("[*] Detected ^GSPC.csv and using it as SP500.")
        else:
            print("[!] Missing SP500 data (expected SP500.csv or ^GSPC.csv).")
            exit(1)

    print("[*] Preparing daily return matrix for all stocks...")

    # Build daily return DataFrame
    return_df = pd.DataFrame({
        symbol: df["Daily Return"] for symbol, df in data.items() if "Daily Return" in df.columns
    })

    return_df.dropna(inplace=True)

    # Get top correlated stocks
    top_stocks = get_top_correlated_stocks(return_df, target_col="SP500", top_n=6)

    # Filter out SP500 or ^GSPC from the feature list
    top_stocks = [s for s in top_stocks if s not in ("SP500", "^GSPC")]

    # Prepare features and target
    X = return_df[top_stocks]
    y = return_df["SP500"]

    # Train model and print metrics
    model, features = build_and_evaluate_model(X, y)

    print("\n📈 Feature Importance (Linear Coefficients):")
    feature_weights = pd.Series(model.coef_, index=features).sort_values(key=abs, ascending=False)
    print(feature_weights)
