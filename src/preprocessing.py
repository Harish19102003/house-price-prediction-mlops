# src/preprocessing.py

import pandas as pd
import numpy as np
import os
import pandas as pd
from sklearn.impute import SimpleImputer

def load_raw_data(filepath: str) -> pd.DataFrame:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"{filepath} does not exist.")
    return pd.read_csv(filepath)

def save_processed_data(df: pd.DataFrame, filepath: str):
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
    num_imputer = SimpleImputer(strategy='mean')
    df[numerical_cols] = num_imputer.fit_transform(df[numerical_cols])
    
    # Impute categorical columns with the most frequent value (mode)
    cat_imputer = SimpleImputer(strategy='most_frequent')
    df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])
    
    return df

