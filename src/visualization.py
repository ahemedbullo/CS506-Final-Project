import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
import pandas as pd
import os
import numpy as np
# --- Added imports implied by the snippet ---
from sklearn.metrics import r2_score # Assuming you use scikit-learn for metrics
# from sklearn.model_selection import train_test_split # Example import for splitting
# from sklearn.linear_model import LinearRegression # Example import for a model
# --- End added imports ---

# Create the data/visualizations directory if it doesn't exist
VIS_DIR = os.path.join("data", "visualizations")
os.makedirs(VIS_DIR, exist_ok=True)

# =============================================================================
#  Plotting Functions (Your existing functions - mostly unchanged)
# =============================================================================

def plot_time_series(df, columns, title="Time Series Plot", ylabel="Price", xlabel="Date", filename="time_series.png"):
    """
    Plots time series of specified columns and saves it as a PNG in data/visualizations/.
    """
    try:
        if df.index.name is None: df.index.name = 'Date'
        if not pd.api.types.is_datetime64_any_dtype(df.index):
            print(f"[*] Converting index to datetime for: {filename}")
            df.index = pd.to_datetime(df.index)
        df = df.sort_index()
    except Exception as e:
        print(f"[!] Failed to convert index to datetime for {title}: {e}. Skipping plot.")
        return

    plt.figure(figsize=(14, 7))
    plot_created = False
    for col in columns:
        if col in df.columns:
            plt.plot(df.index, df[col], label=col)
            plot_created = True
        else:
            print(f"[!] Warning: Column '{col}' not found for plot '{title}'.")

    if not plot_created:
        print(f"[!] No data plotted for {title}. Skipping save.")
        plt.close()
        return

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    save_path = os.path.join(VIS_DIR, filename)
    try:
        plt.savefig(save_path)
        print(f"[+] Saved time series plot to {save_path}")
    except Exception as e:
        print(f"[!] Failed to save plot {save_path}: {e}")
    plt.close()


def plot_correlation_heatmap(df, title="Correlation Heatmap", filename="correlation_heatmap.png"):
    """
    Plots and saves a heatmap of correlations between DataFrame columns.
    Assumes input df contains numerical columns (e.g., daily returns).
    """
    plt.figure(figsize=(16, 12))
    try:
        numeric_df = df.select_dtypes(include=np.number)
        if numeric_df.empty:
            print("[!] No numeric data found for correlation heatmap.")
            plt.close()
            return
        if numeric_df.shape[1] < 2:
            print("[!] Need at least two numeric columns for correlation heatmap.")
            plt.close()
            return

        corr = numeric_df.corr()
        annot_size = max(4, 10 - int(corr.shape[0] / 5))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, annot_kws={"size": annot_size})
        plt.title(title, fontsize=16)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(rotation=0, fontsize=10)
        plt.tight_layout()

        save_path = os.path.join(VIS_DIR, filename)
        plt.savefig(save_path)
        print(f"[+] Saved correlation heatmap to {save_path}")
    except Exception as e:
        print(f"[!] Failed to generate or save correlation heatmap: {e}")
    plt.close()


def plot_scatter_with_regression(x, y, x_label, y_label, title="Scatter Plot with Regression", filename="scatter_regression.png"):
    """
    Plots a scatter plot with a regression line and saves it.
    Input x and y should be pandas Series with potentially different indices.
    """
    if not isinstance(x, pd.Series): x = pd.Series(x)
    if not isinstance(y, pd.Series): y = pd.Series(y)

    aligned_data = pd.concat([x.rename('x'), y.rename('y')], axis=1)
    aligned_data.dropna(inplace=True)

    if aligned_data.empty or len(aligned_data) < 2:
        print(f"[!] Skipping plot '{title}' - Not enough overlapping/valid data points after alignment.")
        return

    plt.figure(figsize=(10, 6))
    try:
        sns.regplot(x=aligned_data['x'], y=aligned_data['y'], scatter_kws={"alpha": 0.5, "s": 15}, line_kws={"color": "red"})
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()

        save_path = os.path.join(VIS_DIR, filename)
        plt.savefig(save_path)
        print(f"[+] Saved scatter regression plot to {save_path}")
    except Exception as e:
        print(f"[!] Failed to generate or save scatter plot '{title}': {e}")
    plt.close()


def plot_predicted_vs_actual(y_true, y_pred, title="Predicted vs Actual S&P 500 Returns", filename="predicted_vs_actual.png"):
    """
    Plots predicted vs actual values and saves the plot.
    Handles potential NaN values and ensures index alignment.
    """
    # Ensure inputs are pandas Series for reliable index alignment
    if not isinstance(y_true, pd.Series):
        print(f"[!] Warning: Converting y_true to Series for plot '{title}'.")
        y_true = pd.Series(y_true, name="Actual")
    if not isinstance(y_pred, pd.Series):
        print(f"[!] Warning: Converting y_pred to Series for plot '{title}'.")
        y_pred = pd.Series(y_pred, name="Predicted")

    # Explicitly align indices BEFORE combining/dropna
    common_index = y_true.index.intersection(y_pred.index)
    if common_index.empty:
        print(f"[!] Skipping plot '{title}' - No common index between y_true and y_pred.")
        return

    y_true_aligned = y_true.loc[common_index]
    y_pred_aligned = y_pred.loc[common_index]

    # Combine aligned series and drop any remaining NaNs
    plot_data = pd.DataFrame({'Actual': y_true_aligned, 'Predicted': y_pred_aligned})
    plot_data.dropna(inplace=True)

    print(f"[*] Data points for plot '{title}' after alignment and dropna: {len(plot_data)}")

    if plot_data.empty:
        print(f"[!] Skipping plot '{title}' - No valid data points remain after dropna.")
        return

    plt.figure(figsize=(10, 6))
    try:
        sns.scatterplot(x=plot_data['Actual'], y=plot_data['Predicted'], alpha=0.6, s=20)
        # Add the y=x line for reference (perfect prediction)
        if len(plot_data) > 1:
            min_val = min(plot_data['Actual'].min(), plot_data['Predicted'].min())
            max_val = max(plot_data['Actual'].max(), plot_data['Predicted'].max())
            # Add padding if min/max are too close or equal
            padding = (max_val - min_val) * 0.05 if max_val > min_val else 0.01
            plt.plot([min_val - padding, max_val + padding], [min_val - padding, max_val + padding], color='red', linestyle='--', label='Perfect Prediction')
            plt.xlim(min_val - padding, max_val + padding)
            plt.ylim(min_val - padding, max_val + padding)
            plt.legend()
        elif len(plot_data) == 1:
             # Handle case with only one data point if necessary (maybe just plot the point)
             pass # Scatter already plots the single point

        plt.xlabel("Actual Daily Return")
        plt.ylabel("Predicted Daily Return")
        plt.title(title)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout() # Use tight_layout before savefig

        save_path = os.path.join(VIS_DIR, filename)
        plt.savefig(save_path)
        print(f"[+] Saved predicted vs actual plot to {save_path}")
    except Exception as e:
        print(f"[!] Failed to generate or save predicted vs actual plot '{title}': {e}")
    plt.close() # Close the plot regardless of success/failure


# =============================================================================
#  Data Loading Function
# =============================================================================

def load_processed_data(data_dir="data/processed"):
    """Loads all processed CSV files from the specified directory."""
    if not os.path.exists(data_dir):
        print(f"[!] Processed data folder '{data_dir}' does not exist!")
        return {}
    data = {}
    print(f"[*] Loading data from: {data_dir}")
    for file in os.listdir(data_dir):
        if file.endswith(".csv"):
            symbol = file.replace(".csv", "")
            file_path = os.path.join(data_dir, file)
            df = None
            try:
                df = pd.read_csv(file_path, index_col="Date", parse_dates=True,
                                 usecols=["Date", "Close", "Daily Return"], # Load required columns
                                 date_format='%Y-%m-%d') # Try specific format first
            except ValueError: # Catch specific error if format fails
                try:
                    print(f"[*] Retrying loading {file} without explicit date_format...")
                    df = pd.read_csv(file_path, index_col="Date", parse_dates=True,
                                     usecols=["Date", "Close", "Daily Return"]) # Fallback
                except Exception as e_retry:
                    print(f"[!] Error loading processed file {file} even on retry: {e_retry}")
                    continue # Skip this file
            except Exception as e:
                print(f"[!] Error loading processed file {file}: {e}")
                continue # Skip this file

            if df is not None and not df.empty:
                if not pd.api.types.is_datetime64_any_dtype(df.index):
                    try:
                        df.index = pd.to_datetime(df.index)
                    except Exception as e_dt:
                        print(f"[!] Failed to convert index for {symbol} to datetime: {e_dt}. Skipping.")
                        continue
                df.sort_index(inplace=True)
                if not df.index.is_monotonic_increasing:
                    print(f"[!] Warning: Index for {symbol} is not monotonic increasing. Sorting again.")
                    df.sort_index(inplace=True)
                data[symbol] = df
            else:
                print(f"[!] Warning: No data loaded for {symbol}.")

    print(f"[+] Loaded data for {len(data)} symbols.")
    return data


# =============================================================================
#  Main Execution Block
# =============================================================================

if __name__ == "__main__":
    print("[*] Loading processed stock data for visualization...")
    data = load_processed_data()
    if not data:
        print("[!] No data loaded, cannot generate visualizations.")
        exit(1)

    # --- Handle S&P 500 alias (remains the same) ---
    sp500_ticker = None
    if "^GSPC" in data:
        sp500_ticker = "^GSPC"
        if "SP500" not in data: data["SP500"] = data["^GSPC"]
        print("[*] Using ^GSPC as S&P 500 data source.")
    elif "SP500" in data:
        sp500_ticker = "SP500"
        print("[*] Using SP500 as S&P 500 data source.")
    else:
        print("[!] Missing S&P 500 data (expected SP500.csv or ^GSPC.csv).")
        sp500_returns = None

    if sp500_ticker:
        sp500_df = data.get(sp500_ticker)
        if sp500_df is not None and "Daily Return" in sp500_df.columns:
            sp500_returns = sp500_df["Daily Return"]
        else:
            print(f"[!] Warning: 'Daily Return' column missing for {sp500_ticker}.")
            sp500_returns = None
    else:
        sp500_returns = None
    # --- End S&P 500 handling ---

    print("\n[*] Generating standard visualizations...")

    # 1. Time series plots for closing prices (remains the same)
    for symbol, df in data.items():
        if "Close" in df.columns:
            plot_time_series(df, ["Close"], f"{symbol} Close Price Over Time", ylabel="Price (USD)", filename=f"{symbol}_time_series.png")
        else:
            print(f"[!] Skipping Close price plot for {symbol} - 'Close' column missing.")

    # 2. Correlation heatmap of daily returns (remains the same)
    return_dict = {}
    for symbol, df in data.items():
        if "Daily Return" in df.columns:
            return_dict[symbol] = df["Daily Return"]
        else:
            print(f"[!] Skipping {symbol} for correlation heatmap - 'Daily Return' column missing.")

    if return_dict:
        return_df = pd.DataFrame(return_dict)
        plot_correlation_heatmap(return_df, "Daily Returns Correlation Heatmap", "daily_returns_correlation.png")
    else:
        print("[!] No data available for correlation heatmap.")

    # 3. Scatter/Regression plots vs S&P 500 (remains the same)
    if sp500_returns is not None:
        print("\n[*] Generating aligned scatter/regression plots vs S&P 500...")
        for symbol, df in data.items():
            if symbol == sp500_ticker or symbol == "SP500":
                continue
            print(f"[*] Plotting regression for {symbol} vs S&P 500...")
            if "Daily Return" in df.columns:
                stock_returns = df["Daily Return"]
                plot_scatter_with_regression(
                    x=stock_returns,
                    y=sp500_returns,
                    x_label=f"{symbol} Daily Return",
                    y_label="S&P 500 Daily Return",
                    title=f"{symbol} vs S&P 500 Daily Returns",
                    filename=f"{symbol}_vs_SP500_regression.png"
                )
            else:
                print(f"[!] Skipping scatter plot for {symbol} - 'Daily Return' column missing.")
    else:
        print("[!] Skipping scatter/regression plots vs S&P 500 - S&P 500 data missing.")


    # =============================================================================
    #  CONCEPTUAL PLACEMENT FOR MODEL EVALUATION AND PREDICTED VS ACTUAL PLOTS
    # =============================================================================
    print("\n[*] === Placeholder: Model Training and Evaluation Section === ")

    # --- !!! This section is hypothetical !!! ---
    # You would need to add your actual ML code here, including:
    # 1. Feature Engineering (creating X from your loaded data)
    # 2. Target Variable Definition (defining y, likely shifted returns)
    # 3. Data Splitting (train_test_split into X_train, X_test, y_train, y_test)
    # 4. Model Definition (e.g., LinearRegression, RandomForestRegressor, etc.)
    # 5. Model Training (model.fit(X_train, y_train))
    # 6. Model Evaluation Loop (if testing multiple models)

    # Example Placeholder Structure:
    models_to_evaluate = {
         #"Linear Regression": LinearRegression(), # Example
         #"Random Forest": RandomForestRegressor(n_estimators=100, random_state=42) # Example
         # Add your instantiated models here
    }
    results_df = pd.DataFrame() # To store predictions if needed
    metrics_summary = {} # To store performance metrics
    LAG_DAYS = 5 # Example Lag - ** YOU NEED TO DEFINE THIS BASED ON YOUR FEATURES **

    # --- Assume X_test, y_test are defined from your split ---
    # X_test = ... # Your test features DataFrame/array
    # y_test = ... # Your test target Series (MUST have a DatetimeIndex matching X_test)

    # Check if y_test exists and has the right index before looping
    if 'y_test' in locals() and isinstance(y_test, pd.Series) and pd.api.types.is_datetime64_any_dtype(y_test.index):
        results_df = pd.DataFrame(index=y_test.index) # Initialize with y_test's index
        results_df['Actual'] = y_test # Store actual values

        for model_name, model in models_to_evaluate.items():
            print(f"\n[*] Evaluating {model_name}...")

            # --- Placeholder for Training (usually done before this point) ---
            # print(f"[*] Assuming {model_name} is already trained...")
            # model.fit(X_train, y_train) # Or load a pre-trained model

            # --- Placeholder for Prediction ---
            # print(f"[*] Predicting on test set with {model_name}...")
            # y_pred = model.predict(X_test) # This would be your actual prediction step

            # --- !!! START OF YOUR SNIPPET INTEGRATION !!! ---
            # --- !!! (Using placeholder y_pred for demonstration) !!! ---
            # --- Create dummy y_pred aligned with y_test for example ---
            np.random.seed(42) # for reproducible dummy data
            y_pred = y_test + np.random.normal(0, y_test.std() * 0.5, size=len(y_test)) # Dummy predictions
            # --- End dummy data ---


            # ... after evaluating model on test set ...
            # y_pred = model.predict(X_test) # <--- YOUR ACTUAL PREDICTION WOULD BE HERE
            test_r2 = r2_score(y_test, y_pred)
            # ... calculate other metrics ...
            # Example:
            # test_mae = mean_absolute_error(y_test, y_pred)
            # test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            print(f"[*] {model_name} Test R-squared: {test_r2:.4f}")

            # Store predictions in results_df
            results_df[f'Predicted_{model_name.replace(" ", "_")}'] = y_pred # Note: This adds potentially unaligned y_pred if not handled carefully

            # Compile metrics dictionary...
            metrics_summary[model_name] = {
                'R-squared': test_r2,
                # 'MAE': test_mae,
                # 'RMSE': test_rmse,
                # Add other metrics...
            }

            # Save model object and performance summary...
            # Example: joblib.dump(model, f'models/{model_name}.pkl')
            # pd.DataFrame(metrics_summary).T.to_csv('results/model_performance.csv')

            # 5e. Generate Predicted vs Actual Plot for this model
            print(f"[*] Generating Predicted vs Actual plot for {model_name}...")
            if callable(plot_predicted_vs_actual):
                # --- FIX: Convert y_pred array to Series with y_test's index ---
                # This ensures the plotting function receives correctly aligned data
                # y_pred is often a numpy array from model.predict()
                y_pred_series = pd.Series(y_pred, index=y_test.index, name="Predicted")
                # ---------------------------------------------------------------

                base_plot_filename = f"predicted_vs_actual_{model_name.lower().replace(' ', '_')}.png"
                plot_predicted_vs_actual(
                    y_true=y_test,        # Pass original y_test Series (has index)
                    y_pred=y_pred_series, # Pass the new y_pred Series with index
                    title=f"Predicted vs Actual S&P 500 Returns ({model_name} - Lag {LAG_DAYS}d)",
                    filename=base_plot_filename
                )
            else:
                print("[!] `plot_predicted_vs_actual` function not available for plotting.")
            # --- !!! END OF YOUR SNIPPET INTEGRATION !!! ---

    else:
        print("[!] Skipping model evaluation plots: 'y_test' not defined or not a Series with DatetimeIndex.")

    # --- End Placeholder Section ---

    print("\n✅ Visualization and Conceptual ML Evaluation script finished.")