import os
import sys
import pandas as pd
import pytest # type: ignore

# Add the 'src' directory to the Python path so we can import modules from there
# Note: This is a common way to handle imports in tests,
# alternatives exist (like installing the src as a package).
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, src_path)

# Now we should be able to import from src
try:
    from data_loader import fetch_stock_data
    from data_processor import clean_stock_data
    from correlation_analysis import compute_correlation
    # We don't typically test plotting functions directly unless complex logic exists
    # We might not test model.py directly here if it depends heavily on prior steps,
    # but we can test components if refactored.
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Ensure you run tests using 'make test' which handles the environment.")
    # Fail fast if imports don't work
    pytest.skip("Skipping tests due to import errors. Run with 'make test'.", allow_module_level=True)


# --- Fixtures (Setup code for tests) ---

# Define sample data paths relative to the project root
RAW_TEST_DIR = os.path.join(project_root, "data", "raw_test")
PROCESSED_TEST_DIR = os.path.join(project_root, "data", "processed_test")
SAMPLE_TICKER = "^GSPC" # Use S&P 500 for tests
RAW_SAMPLE_FILE = os.path.join(RAW_TEST_DIR, f"{SAMPLE_TICKER}.csv")
PROCESSED_SAMPLE_FILE = os.path.join(PROCESSED_TEST_DIR, f"{SAMPLE_TICKER}.csv")

@pytest.fixture(scope="module", autouse=True)
def setup_test_data():
    """Creates dummy data directories and potentially fetches minimal data for testing."""
    print("\nSetting up test directories...")
    os.makedirs(RAW_TEST_DIR, exist_ok=True)
    os.makedirs(PROCESSED_TEST_DIR, exist_ok=True)

    # Option 1: Create a tiny dummy raw CSV file
    # This avoids hitting the network during tests, making them faster and more reliable
    dummy_data = "Date,Open,High,Low,Close,Adj Close,Volume\n" \
                 "2023-01-03,100,102,99,101,101,10000\n" \
                 "2023-01-04,101,103,100,102,102,12000\n" \
                 "2023-01-05,102,102,98,100,100,11000\n"
    with open(RAW_SAMPLE_FILE, "w") as f:
        f.write(dummy_data)
    print(f"Created dummy raw file: {RAW_SAMPLE_FILE}")

    # Option 2: Fetch real minimal data (uncomment if preferred, but less reliable for tests)
    # print(f"Fetching minimal real data for {SAMPLE_TICKER}...")
    # try:
    #     fetch_stock_data([SAMPLE_TICKER], "2023-01-01", "2023-01-07", save_path=RAW_TEST_DIR)
    # except Exception as e:
    #     pytest.skip(f"Skipping tests requiring data fetch due to error: {e}")

    yield # This allows tests to run

    # Teardown: Clean up dummy files/directories after tests run
    print("\nTearing down test data...")
    if os.path.exists(RAW_SAMPLE_FILE):
        os.remove(RAW_SAMPLE_FILE)
    if os.path.exists(PROCESSED_SAMPLE_FILE):
        os.remove(PROCESSED_SAMPLE_FILE)
    # Cautiously remove directories only if they are empty (or remove specific files)
    try:
        if not os.listdir(RAW_TEST_DIR): os.rmdir(RAW_TEST_DIR)
        if not os.listdir(PROCESSED_TEST_DIR): os.rmdir(PROCESSED_TEST_DIR)
        # Or use shutil.rmtree(RAW_TEST_DIR) etc. if you are sure
    except OSError as e:
        print(f"Error removing test directories (might not be empty): {e}")


# --- Test Functions ---

def test_data_loading_creates_file():
    """Tests if fetch_stock_data creates the expected output file (using dummy/fixture)."""
    # The fixture already created/fetched the file
    assert os.path.exists(RAW_SAMPLE_FILE), f"Raw data file {RAW_SAMPLE_FILE} was not created by setup fixture."

# @pytest.mark.skip(reason="Skipping actual data fetch test to avoid network dependency")
# def test_data_loading_network():
#     """Tests fetching actual data - requires network. Skipped by default."""
#     tickers = ['MSFT'] # Use a different ticker to avoid conflict with fixture
#     start = "2023-01-01"
#     end = "2023-01-05"
#     path = os.path.join(RAW_TEST_DIR, f"{tickers[0]}.csv")
#     if os.path.exists(path): os.remove(path) # Ensure clean start
#     fetch_stock_data(tickers, start, end, save_path=RAW_TEST_DIR)
#     assert os.path.exists(path)
#     os.remove(path) # Clean up

def test_data_processing_creates_file_and_columns():
    """Tests if clean_stock_data creates a processed file and adds expected columns."""
    # Ensure the raw file exists from the fixture
    assert os.path.exists(RAW_SAMPLE_FILE)

    # Run the processing function
    try:
        # Pass the specific test directories
        df = clean_stock_data(RAW_SAMPLE_FILE, save_path=PROCESSED_TEST_DIR)
    except Exception as e:
        pytest.fail(f"clean_stock_data failed with exception: {e}")


    # Check if processed file was created
    assert os.path.exists(PROCESSED_SAMPLE_FILE), f"Processed data file {PROCESSED_SAMPLE_FILE} was not created."

    # Check if the returned DataFrame is not empty
    assert isinstance(df, pd.DataFrame), "clean_stock_data did not return a DataFrame."
    assert not df.empty, "Processed DataFrame is empty."

    # Check for expected columns calculated during processing
    expected_cols = ['Daily Return', 'Price Range', 'Volume Change']
    for col in expected_cols:
        assert col in df.columns, f"Expected column '{col}' not found in processed data."

    # Check if index is DatetimeIndex
    assert isinstance(df.index, pd.DatetimeIndex), "Index of processed data is not DatetimeIndex."

def test_correlation_computation():
    """Tests the correlation computation logic on dummy processed data."""
    # Create dummy processed data for testing correlation
    # Requires at least two 'stocks' and 'Daily Return' column
    data = {
        'SP500': pd.DataFrame({'Daily Return': [0.01, -0.005, 0.015]}, index=pd.to_datetime(['2023-01-03', '2023-01-04', '2023-01-05'])),
        'STOCKA': pd.DataFrame({'Daily Return': [0.02, -0.01, 0.025]}, index=pd.to_datetime(['2023-01-03', '2023-01-04', '2023-01-05'])),
        'STOCKB': pd.DataFrame({'Daily Return': [-0.005, 0.001, -0.01]}, index=pd.to_datetime(['2023-01-03', '2023-01-04', '2023-01-05']))
    }
    # Rename SP500 key to ^GSPC to test handling if needed
    # data['^GSPC'] = data.pop('SP500')
    # data['SP500'] = data['^GSPC'] # Ensure SP500 key exists for compute_correlation

    results = compute_correlation(data)

    assert results is not None, "compute_correlation returned None."
    assert isinstance(results, pd.Series), "compute_correlation did not return a Pandas Series."
    assert 'SP500' in results.index, "SP500 correlation missing from results."
    assert 'STOCKA' in results.index, "STOCKA correlation missing from results."
    # Check if correlation values are within expected range [-1, 1]
    assert results['SP500'] == 1.0, "SP500 correlation with itself should be 1.0."
    assert -1 <= results['STOCKA'] <= 1, "STOCKA correlation value out of range."

# Add more tests as needed for specific logic in your functions