import pandas as pd
import os

def clean_stock_data(file_path, save_path="data/processed/"):
    """Load, clean, and process stock data."""
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    try:
        # Skip the first two rows (Ticker and Date rows) and use the third row as headers
        df = pd.read_csv(file_path, skiprows=[1])
        
        print(f"\nProcessing file: {file_path}")
        print("Columns in CSV:", df.columns.tolist())
        
        # Convert string values to numeric, coercing errors to NaN
        numeric_columns = ['Close', 'High', 'Low', 'Open']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Set the index to Date
        df.set_index('Price', inplace=True)
        df.index.name = 'Date'
        
        print("\nFirst few rows after processing:")
        print(df.head())

        # Clean and process data
        df.dropna(inplace=True)
        
        # Calculate daily returns using Close price
        df["Daily Return"] = df["Close"].pct_change()

        # Save processed data
        cleaned_file_path = os.path.join(save_path, os.path.basename(file_path))
        df.to_csv(cleaned_file_path)
        print(f"\nSaved processed file to: {cleaned_file_path}")
        
        return df
    
    except Exception as e:
        print(f"\nError processing {file_path}: {str(e)}")
        raise

if __name__ == "__main__":
    raw_data_path = "data/raw/"
    processed_data_path = "data/processed/"

    # Make sure the raw data directory exists
    if not os.path.exists(raw_data_path):
        print(f"Error: Raw data directory '{raw_data_path}' does not exist!")
        exit(1)

    # Process each CSV file
    files = os.listdir(raw_data_path)
    if not files:
        print(f"No files found in {raw_data_path}")
    else:
        print(f"\nProcessing files from {raw_data_path}")
        print(f"Processed files will be saved to {processed_data_path}")
        
        for file in files:
            if file.endswith(".csv"):
                print(f"\nProcessing {file}...")
                try:
                    clean_stock_data(os.path.join(raw_data_path, file))
                    print(f"Successfully processed {file}")
                except Exception as e:
                    print(f"Failed to process {file}: {str(e)}")
                    continue