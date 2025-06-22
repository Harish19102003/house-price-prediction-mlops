# src/preprocessing.py

import pandas as pd
import numpy as np
import os
from typing import Optional, List
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from pathlib import Path

def load_raw_data(filepath: str) -> pd.DataFrame:
    """Load raw data from CSV file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"{filepath} does not exist.")
    return pd.read_csv(filepath)

def save_processed_data(df: pd.DataFrame, filepath: str):
    """Save processed data to CSV file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)

def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fills missing values in the dataframe.
    - Numerical columns will be filled with the mean or median value.
    - Categorical columns will be filled with the most frequent value (mode).
    
    Parameters:
        df: The dataframe containing missing values.
        
    Returns:
        df: The dataframe with missing values filled.
    """
    
    # Identify numerical and categorical columns
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    # Impute numerical columns with the mean or median
    if len(numerical_cols) > 0:
        num_imputer = SimpleImputer(strategy='mean')
        df[numerical_cols] = num_imputer.fit_transform(df[numerical_cols])
    
    # Impute categorical columns with the most frequent value (mode)
    if len(categorical_cols) > 0:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])
    
    return df

def encode_categorical_features(df: pd.DataFrame, target_column: str = 'SalePrice', 
                              training_columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Encode categorical features for machine learning.
    Uses Label Encoding for ordinal features and One-Hot Encoding for nominal features.
    
    Parameters:
        df: The dataframe with categorical features.
        target_column: Name of the target column to exclude from encoding.
        training_columns: List of expected column names from training (for inference).
        
    Returns:
        df: The dataframe with encoded categorical features.
    """
    
    df_encoded = df.copy()
    
    # Identify categorical columns (excluding target)
    categorical_cols = df_encoded.select_dtypes(include=['object']).columns.tolist()
    if target_column in categorical_cols:
        categorical_cols.remove(target_column)
    
    # Define ordinal features that have inherent order
    ordinal_features = {
        'LotShape': ['Reg', 'IR1', 'IR2', 'IR3'],
        'Utilities': ['AllPub', 'NoSewr', 'NoSeWa', 'ELO'],
        'LandSlope': ['Gtl', 'Mod', 'Sev'],
        'ExterQual': ['Ex', 'Gd', 'TA', 'Fa', 'Po'],
        'ExterCond': ['Ex', 'Gd', 'TA', 'Fa', 'Po'],
        'BsmtQual': ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA'],
        'BsmtCond': ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA'],
        'BsmtExposure': ['Gd', 'Av', 'Mn', 'No', 'NA'],
        'BsmtFinType1': ['GLQ', 'ALQ', 'BLQ', 'Rec', 'LwQ', 'Unf', 'NA'],
        'BsmtFinType2': ['GLQ', 'ALQ', 'BLQ', 'Rec', 'LwQ', 'Unf', 'NA'],
        'HeatingQC': ['Ex', 'Gd', 'TA', 'Fa', 'Po'],
        'KitchenQual': ['Ex', 'Gd', 'TA', 'Fa', 'Po'],
        'Functional': ['Typ', 'Min1', 'Min2', 'Mod', 'Maj1', 'Maj2', 'Sev', 'Sal'],
        'FireplaceQu': ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA'],
        'GarageQual': ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA'],
        'GarageCond': ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA'],
        'PoolQC': ['Ex', 'Gd', 'TA', 'Fa', 'NA'],
        'Fence': ['GdPrv', 'MnPrv', 'GdWo', 'MnWw', 'NA']
    }
    
    # Apply ordinal encoding
    for col in categorical_cols:
        if col in ordinal_features and col in df_encoded.columns:
            # Create mapping for ordinal features
            order_mapping = {val: idx for idx, val in enumerate(ordinal_features[col])}
            # Handle any values not in the predefined order
            unique_vals = df_encoded[col].unique()
            for val in unique_vals:
                if val not in order_mapping:
                    order_mapping[val] = len(order_mapping)
            
            df_encoded[col] = df_encoded[col].map(order_mapping)
    
    # Apply one-hot encoding to remaining categorical features
    remaining_categorical = [col for col in categorical_cols 
                           if col not in ordinal_features and col in df_encoded.columns]
    
    if remaining_categorical:
        # Use pandas get_dummies for simplicity
        df_encoded = pd.get_dummies(df_encoded, columns=remaining_categorical, 
                                  prefix=remaining_categorical, drop_first=True)
    
    # If we have training columns, ensure we have the same columns
    if training_columns is not None:
        # Add missing columns with zeros - use a more efficient approach
        missing_cols = set(training_columns) - set(df_encoded.columns)
        if missing_cols:
            # Create a DataFrame with missing columns and concatenate
            missing_df = pd.DataFrame(0, index=df_encoded.index, columns=list(missing_cols))
            df_encoded = pd.concat([df_encoded, missing_df], axis=1)
        
        # Remove extra columns that weren't in training
        extra_cols = set(df_encoded.columns) - set(training_columns)
        if extra_cols:
            print(f"Removing extra columns not in training: {extra_cols}")
            df_encoded = df_encoded.drop(columns=list(extra_cols))
        
        # Reorder columns to match training exactly
        df_encoded = df_encoded[training_columns]
    
    return df_encoded

def preprocess_data(df: pd.DataFrame, target_column: str = 'SalePrice', 
                   training_columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Complete preprocessing pipeline:
    1. Fill missing values
    2. Encode categorical features
    3. Handle any remaining data issues
    
    Parameters:
        df: The raw dataframe.
        target_column: Name of the target column.
        training_columns: List of expected column names from training (for inference).
        
    Returns:
        df_processed: The fully processed dataframe ready for ML.
    """
    print(f"Starting preprocessing with shape: {df.shape}")
    
    # Step 1: Fill missing values
    df_filled = fill_missing_values(df.copy())
    print(f"After filling missing values: {df_filled.shape}")
    
    # Step 2: Encode categorical features
    df_encoded = encode_categorical_features(df_filled, target_column, training_columns)
    print(f"After encoding categorical features: {df_encoded.shape}")
    
    # Step 3: Convert all columns to numeric (except target if it's categorical)
    for col in df_encoded.columns:
        if col != target_column:
            df_encoded[col] = pd.to_numeric(df_encoded[col], errors='coerce')
    
    # Step 4: Handle any remaining missing values created during conversion
    df_final = df_encoded.fillna(0)
    
    print(f"Final processed shape: {df_final.shape}")
    print(f"Data types: {df_final.dtypes.value_counts()}")
    
    return df_final

