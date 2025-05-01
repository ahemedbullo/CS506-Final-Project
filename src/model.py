# src/model.py

import pandas as pd
import numpy as np
import os
import joblib # For saving the model
from sklearn.linear_model import LinearRegression # type: ignore
from sklearn.ensemble import RandomForestRegressor # type: ignore # Added for comparison
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score # type: ignore # Added TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score # type: ignore

# Assuming visualization.py is in the same src directory
try:
    from visualization import plot_predicted_vs_actual
except ImportError:
    print("[!] Warning: visualization.py not found or plot_predicted_vs_actual not importable.")
    # Define a dummy function if import fails, so the script doesn't crash
    def plot_predicted_vs_actual(*args, **kwargs):
        print("[!] Skipping predicted vs actual plot due to import error.")

# --- Configuration ---
PROCESSED_DATA_DIR = os.path.join("data", "processed")
CORRELATION_FILE = "correlation_results.csv" # Expect correlation results in root dir
RESULTS_DIR = "results" # Save outputs here
MODELS_DIR = "models" # Save trained models here
MODEL_RESULTS_CSV = os.path.join(RESULTS_DIR, "model_performance_results.csv")
MODEL_PERFORMANCE_SUMMARY = os.path.join(RESULTS_DIR, "model_performance_summary.txt")
MODEL_SAVE_PATH_LR = os.path.join(MODELS_DIR, "linear_regression_model.joblib") # Paths for saved models
MODEL_SAVE_PATH_RF = os.path.join(MODELS_DIR, "random_forest_model.joblib")

N_TOP_FEATURES = 5 # Number of top correlated stocks to use as features
LAG_DAYS = 1 # Number of days to lag features (e.g., 1 means use day t-1 to predict day t)
TEST_SIZE = 0.2 # 80% train, 20% test
CV_SPLITS = 5 # Number of splits for TimeSeriesSplit cross-validation
RANDOM_STATE = 42 # For reproducibility in models like RandomForest

# --- Helper Functions ---

def load_processed_data(data_dir):
    """Loads 'Daily Return' from processed CSV files."""
    if not os.path.exists(data_dir):
        print(f"[!] Error: Processed data directory '{data_dir}' not found.")
        return None
    data = {}
    print(f"[*] Loading data from: {data_dir}")
    for file in os.listdir(data_dir):
        if file.endswith(".csv"):
            try:
                symbol = file.replace(".csv", "")
                file_path = os.path.join(data_dir, file)
                # Only load 'Daily Return', ensure index is datetime
                df = pd.read_csv(file_path, index_col="Date", parse_dates=True, usecols=["Date", "Daily Return"])
                df.sort_index(inplace=True)
                if not df.index.is_monotonic_increasing:
                     print(f"[!] Warning: Index for {symbol} is not monotonic increasing. Sorting again.")
                     df.sort_index(inplace=True)
                if df['Daily Return'].isnull().any():
                     print(f"[!] Warning: NaNs found in 'Daily Return' for {symbol}. Will be handled during merge.")
                data[symbol] = df['Daily Return']
            except Exception as e:
                print(f"[!] Error loading processed file {file}: {e}")
    print(f"[+] Loaded data for {len(data)} symbols.")
    return data

def select_top_features(correlation_file, n_top, target_symbol):
    """Selects top N features based on absolute correlation with the target."""
    try:
        corr_df = pd.read_csv(correlation_file)
        # Ensure target_symbol is in the 'Stock' column for exclusion later
        if target_symbol not in corr_df['Stock'].values:
             print(f"[!] Warning: Target symbol '{target_symbol}' not found in correlation file's 'Stock' column.")
             # Attempt to find common variations like ^GSPC if SP500 is target
             if target_symbol == 'SP500' and '^GSPC' in corr_df['Stock'].values:
                  target_symbol_in_file = '^GSPC'
                  print(f"[*] Using '{target_symbol_in_file}' found in correlation file.")
             elif target_symbol == '^GSPC' and 'SP500' in corr_df['Stock'].values:
                  target_symbol_in_file = 'SP500'
                  print(f"[*] Using '{target_symbol_in_file}' found in correlation file.")
             else:
                  print(f"[!] Cannot determine target symbol in correlation file. Check {correlation_file}")
                  return None
        else:
            target_symbol_in_file = target_symbol

        corr_df['Abs Correlation'] = corr_df['Correlation with SP500'].abs()
        # Exclude the target symbol itself and sort
        corr_df_filtered = corr_df[corr_df['Stock'] != target_symbol_in_file].sort_values(by='Abs Correlation', ascending=False)

        if corr_df_filtered.empty:
            print("[!] Error: No features left after excluding target symbol in correlation file.")
            return None

        top_features = corr_df_filtered['Stock'].head(n_top).tolist()
        print(f"[*] Selected top {n_top} features based on absolute correlation: {top_features}")
        return top_features
    except FileNotFoundError:
        print(f"[!] Error: Correlation file '{correlation_file}' not found. Run 'make correlate' first.")
        return None
    except Exception as e:
        print(f"[!] Error reading or processing correlation file '{correlation_file}': {e}")
        return None

def prepare_lagged_features(data, target_symbol, feature_symbols, lag_days):
    """Prepares features (lagged) and target variable."""
    print(f"[*] Preparing lagged features (lag={lag_days} days)...")
    if target_symbol not in data:
        print(f"[!] Error: Target symbol '{target_symbol}' not found in loaded data.")
        return None, None, None
    missing_features = [f for f in feature_symbols if f not in data]
    if missing_features:
        print(f"[!] Error: Feature symbols not found in loaded data: {missing_features}")
        print(f"Available data keys: {list(data.keys())}")
        return None, None, None

    # Combine target and features into one DataFrame for easier shifting and alignment
    all_symbols = [target_symbol] + feature_symbols
    combined_df = pd.DataFrame({symbol: data[symbol] for symbol in all_symbols})

    # Create target variable y (no shift)
    y = combined_df[target_symbol]

    # Create lagged features X
    X_lagged = combined_df[feature_symbols].shift(lag_days)

    # Combine y and lagged X for proper alignment and NaN dropping
    final_df = pd.concat([y.rename('Target'), X_lagged], axis=1)

    # Drop rows with NaNs introduced by lagging or initial NaNs
    initial_rows = len(final_df)
    final_df.dropna(inplace=True)
    rows_after_na = len(final_df)
    print(f"[*] Dropped {initial_rows - rows_after_na} rows due to NaNs (lagging or merging).")

    if final_df.empty:
        print("[!] Error: No data remaining after applying lag and dropping NaNs.")
        return None, None, None

    # Separate final X and y
    X = final_df[feature_symbols]
    y_final = final_df['Target']
    idx = final_df.index # Keep track of the dates for the final aligned data

    print(f"[*] Lagged data prepared: X shape {X.shape}, y shape {y_final.shape}, Index length {len(idx)}")
    return X, y_final, idx


def calculate_persistence_baseline(y_test):
    """Calculates a persistence baseline (predict t = value at t-1)."""
    # Shift the *actual* test values by 1 day forward to get the prediction for the next day
    # The first prediction will be NaN, needs to be handled in evaluation
    y_pred_baseline = y_test.shift(1)
    # Simple approach: Evaluate only where prediction is not NaN
    valid_indices = ~y_pred_baseline.isnull()
    if not valid_indices.any():
        print("[!] Warning: Could not calculate persistence baseline (no valid shifted values).")
        return None, None, None, {}

    y_test_valid = y_test[valid_indices]
    y_pred_baseline_valid = y_pred_baseline[valid_indices]

    base_r2 = r2_score(y_test_valid, y_pred_baseline_valid)
    base_mae = mean_absolute_error(y_test_valid, y_pred_baseline_valid)
    base_rmse = np.sqrt(mean_squared_error(y_test_valid, y_pred_baseline_valid))
    baseline_metrics = {'Baseline R2': base_r2, 'Baseline MAE': base_mae, 'Baseline RMSE': base_rmse}
    print(f"[*] Persistence Baseline Test R^2: {base_r2:.4f}")
    print(f"[*] Persistence Baseline Test MAE: {base_mae:.6f}")
    print(f"[*] Persistence Baseline Test RMSE: {base_rmse:.6f}")

    # Return full baseline prediction series (with NaN) for results DF, and metrics
    return y_pred_baseline, y_test_valid.index, baseline_metrics


def save_results_and_model(model, model_name, metrics, baseline_metrics, results_df, model_save_path):
    """Saves model, performance metrics, and prediction results."""
    print(f"[*] Saving results for {model_name}...")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    # 1. Save the trained model object
    try:
        joblib.dump(model, model_save_path)
        print(f"[+] Saved {model_name} model to {model_save_path}")
    except Exception as e:
        print(f"[!] Error saving {model_name} model: {e}")

    # 2. Save detailed predictions CSV (potentially append if exists?) - For simplicity, overwrite now
    # We save results only once after all models are evaluated
    # This function focuses on saving the model object and performance summary

    # 3. Save summary performance metrics to a text file (append mode)
    try:
        mode = 'a' if os.path.exists(MODEL_PERFORMANCE_SUMMARY) else 'w'
        with open(MODEL_PERFORMANCE_SUMMARY, mode) as f:
             if mode == 'w': # Write header only if file is new
                  f.write("Model Performance Summary:\n")
                  f.write("=========================\n")
             f.write(f"\n--- {model_name} ---\n")
             for key, value in metrics.items():
                 f.write(f"  - {key}: {value:.4f}\n")

             # Write baseline metrics only once, perhaps after the first model
             if baseline_metrics is not None and model_name == "Linear Regression": # Example: write baseline with LR
                 f.write("\n--- Persistence Baseline ---\n")
                 if baseline_metrics:
                     for key, value in baseline_metrics.items():
                         f.write(f"  - {key}: {value:.4f}\n")
                 else:
                     f.write("  - Baseline metrics not available.\n")

        print(f"[+] Appended {model_name} performance summary to {MODEL_PERFORMANCE_SUMMARY}")
    except Exception as e:
        print(f"[!] Error saving performance summary: {e}")

def save_detailed_results(results_df):
     """Saves the combined results dataframe to CSV."""
     try:
        results_df.to_csv(MODEL_RESULTS_CSV, index=False)
        print(f"[+] Saved detailed prediction results to {MODEL_RESULTS_CSV}")
     except Exception as e:
        print(f"[!] Error saving detailed results CSV: {e}")


# --- Main Execution ---

if __name__ == "__main__":
    print("[*] Starting S&P 500 Forecasting Model Training...")

    # Ensure output directories exist
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    # Clear summary file at the start of a run
    if os.path.exists(MODEL_PERFORMANCE_SUMMARY):
        os.remove(MODEL_PERFORMANCE_SUMMARY)

    # 1. Load Processed Data
    all_returns_data = load_processed_data(PROCESSED_DATA_DIR)
    if not all_returns_data:
        print("[!] No processed data loaded. Exiting.")
        exit(1)

    # Find S&P 500 ticker in loaded data
    sp500_ticker = None
    potential_tickers = ['^GSPC', 'SP500'] # Add other variations if needed
    for ticker in potential_tickers:
        if ticker in all_returns_data:
            sp500_ticker = ticker
            print(f"[*] Using '{sp500_ticker}' as the S&P 500 target ticker.")
            break
    if sp500_ticker is None:
        print(f"[!] Error: S&P 500 data ({' or '.join(potential_tickers)}) not found in loaded data.")
        exit(1)

    # 2. Select Top Features Dynamically (based on contemporaneous correlation)
    print(f"\n[*] Selecting top {N_TOP_FEATURES} features based on correlation file '{CORRELATION_FILE}'...")
    feature_symbols = select_top_features(CORRELATION_FILE, N_TOP_FEATURES, sp500_ticker)
    if feature_symbols is None:
        print("[!] Failed to select features. Exiting.")
        exit(1)

    # 3. Prepare Lagged Data for Modeling
    X, y, idx = prepare_lagged_features(all_returns_data, sp500_ticker, feature_symbols, LAG_DAYS)
    if X is None or y is None:
        print("[!] Failed to prepare lagged data. Exiting.")
        exit(1)

    # 4. Split Data using TimeSeries approach (No Shuffle)
    print(f"\n[*] Splitting data into training ({1-TEST_SIZE:.0%}) and testing ({TEST_SIZE:.0%}) sets (time-ordered)...")
    # We split X, y, and the index idx together
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, idx, test_size=TEST_SIZE, shuffle=False
    )
    if X_train.empty or X_test.empty:
         print("[!] Error: Training or testing set is empty after split.")
         exit(1)
    print(f"[*] Training set size: {X_train.shape[0]}, Testing set size: {X_test.shape[0]}")

    # Create a DataFrame to store results, indexed by test set dates
    results_df = pd.DataFrame({'Actual': y_test}, index=idx_test)

    # --- Baseline Calculation (Persistence) ---
    print("\n[*] Calculating Persistence Baseline...")
    # Use y_test (a Pandas Series) directly
    y_pred_baseline, baseline_valid_idx, baseline_metrics = calculate_persistence_baseline(results_df['Actual'])
    if y_pred_baseline is not None:
        # Add baseline predictions to the results dataframe, aligning by index
        results_df['Baseline_Persistence'] = y_pred_baseline


    # --- Model Training and Evaluation ---
    models_to_evaluate = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1, max_depth=10) # Added RF
    }
    model_save_paths = {
        "Linear Regression": MODEL_SAVE_PATH_LR,
        "Random Forest": MODEL_SAVE_PATH_RF
    }

    # Initialize TimeSeriesSplit for cross-validation
    tscv = TimeSeriesSplit(n_splits=CV_SPLITS)
    print(f"\n[*] Using TimeSeriesSplit with {CV_SPLITS} splits for Cross-Validation.")

    for model_name, model in models_to_evaluate.items():
        print(f"\n--- Training and Evaluating {model_name} ---")

        # 5a. Cross-Validation on Training Set
        print(f"[*] Performing {CV_SPLITS}-fold TimeSeries Cross-Validation...")
        cv_scores_r2 = cross_val_score(model, X_train, y_train, cv=tscv, scoring='r2', n_jobs=-1)
        cv_scores_neg_mae = cross_val_score(model, X_train, y_train, cv=tscv, scoring='neg_mean_absolute_error', n_jobs=-1)
        cv_scores_neg_rmse = cross_val_score(model, X_train, y_train, cv=tscv, scoring='neg_root_mean_squared_error', n_jobs=-1)

        print(f"[*] CV R^2 (Train): {np.mean(cv_scores_r2):.4f} ± {np.std(cv_scores_r2):.4f}")
        print(f"[*] CV MAE (Train): {-np.mean(cv_scores_neg_mae):.6f} ± {np.std(cv_scores_neg_mae):.6f}")
        print(f"[*] CV RMSE (Train): {-np.mean(cv_scores_neg_rmse):.6f} ± {np.std(cv_scores_neg_rmse):.6f}")

        # 5b. Train Final Model on Full Training Set
        print(f"[*] Training final {model_name} model on full training set...")
        model.fit(X_train, y_train)
        print("[+] Training complete.")
        if model_name == "Linear Regression":
            print(f"[*] Coefficients: {model.coef_}")
            print(f"[*] Intercept: {model.intercept_}")

        # 5c. Evaluate on Test Set
        print(f"[*] Evaluating {model_name} on test set...")
        y_pred = model.predict(X_test)
        test_r2 = r2_score(y_test, y_pred)
        test_mae = mean_absolute_error(y_test, y_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred)) # RMSE from MSE

        print("[+] Test Set Evaluation Results:")
        print(f"  - R-squared (R²): {test_r2:.4f}")
        print(f"  - Mean Absolute Error (MAE): {test_mae:.6f}")
        print(f"  - Root Mean Squared Error (RMSE): {test_rmse:.6f}")

        # Store predictions
        results_df[f'Predicted_{model_name.replace(" ", "_")}'] = y_pred

        # Compile metrics
        metrics = {
            'CV R2 Mean (Train)': np.mean(cv_scores_r2),
            'CV R2 Std (Train)': np.std(cv_scores_r2),
            'CV MAE Mean (Train)': -np.mean(cv_scores_neg_mae),
            'CV RMSE Mean (Train)': -np.mean(cv_scores_neg_rmse),
            'Test R2': test_r2,
            'Test MAE': test_mae,
            'Test RMSE': test_rmse
        }

        # 5d. Save model object and performance summary
        save_results_and_model(
            model=model,
            model_name=model_name,
            metrics=metrics,
            baseline_metrics=baseline_metrics if model_name == "Linear Regression" else None, # Only save baseline metrics once
            results_df=results_df, # Pass results_df (although not used in save func currently)
            model_save_path=model_save_paths[model_name]
            )

        # 5e. Generate Predicted vs Actual Plot for this model
        print(f"[*] Generating Predicted vs Actual plot for {model_name}...")
        if callable(plot_predicted_vs_actual):
            plot_filename = os.path.join(VIS_DIR, f"predicted_vs_actual_{model_name.lower().replace(' ', '_')}.png")
            plot_predicted_vs_actual(
                y_true=y_test,
                y_pred=y_pred,
                title=f"Predicted vs Actual S&P 500 Returns ({model_name} - Lag {LAG_DAYS}d)",
                filename=plot_filename # Pass the full path
            )
        else:
            print("[!] `plot_predicted_vs_actual` function not available for plotting.")


    # 6. Save Combined Detailed Results
    print(f"\n[*] Saving combined detailed results to {MODEL_RESULTS_CSV}...")
    results_df.reset_index(inplace=True) # Make Date a column
    save_detailed_results(results_df)

    print("\n[*] Model script finished successfully.")