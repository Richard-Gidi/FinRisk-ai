"""
Data loading and preprocessing utilities for the FinRisk-ai project.
"""

import pandas as pd
from pathlib import Path

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file with proper type inference.
    
    Args:
        file_path (str): Path to the CSV file
    
    Returns:
        pd.DataFrame: Loaded and preprocessed DataFrame
    """
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error loading data from {file_path}: {str(e)}")
        return None

def preprocess_dates(df: pd.DataFrame, date_columns: list) -> pd.DataFrame:
    """
    Convert date columns to datetime format.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        date_columns (list): List of column names containing dates
    
    Returns:
        pd.DataFrame: DataFrame with processed date columns
    """
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    return df

def clean_numeric(df: pd.DataFrame, numeric_columns: list) -> pd.DataFrame:
    """
    Clean numeric columns by handling missing values and outliers.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        numeric_columns (list): List of numeric column names
    
    Returns:
        pd.DataFrame: DataFrame with cleaned numeric columns
    """
    for col in numeric_columns:
        if col in df.columns:
            # Fill missing values with median
            df[col] = df[col].fillna(df[col].median())
            
            # Handle outliers (e.g., cap at 99th percentile)
            upper_limit = df[col].quantile(0.99)
            df[col] = df[col].clip(upper=upper_limit)
    
    return df
