#!/usr/bin/env python
# Script to reprocess data with proper categorical encoding

from src.preprocessing import load_raw_data, preprocess_data, save_processed_data

def main():
    print("🔄 Reprocessing data with categorical encoding...")
    
    # Load raw data
    raw_data_path = "data/raw/train.csv"
    print(f"Loading raw data from: {raw_data_path}")
    df = load_raw_data(raw_data_path)
    print(f"Raw data shape: {df.shape}")
    
    # Apply complete preprocessing
    df_processed = preprocess_data(df, target_column='SalePrice')
    
    # Save processed data
    processed_data_path = "data/processed/train_processed.csv"
    print(f"Saving processed data to: {processed_data_path}")
    save_processed_data(df_processed, processed_data_path)
    
    print("✅ Data reprocessing completed successfully!")
    print(f"Final shape: {df_processed.shape}")
    print(f"Columns: {len(df_processed.columns)}")
    
    # Show data types
    numeric_cols = df_processed.select_dtypes(include=['int64', 'float64']).columns
    object_cols = df_processed.select_dtypes(include=['object']).columns
    
    print(f"Numeric columns: {len(numeric_cols)}")
    print(f"Object columns: {len(object_cols)}")
    
    if len(object_cols) > 0:
        print(f"Remaining object columns: {list(object_cols)}")

if __name__ == "__main__":
    main()
