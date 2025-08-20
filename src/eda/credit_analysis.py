"""
Credit analysis utilities for risk assessment.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

class CreditAnalyzer:
    """
    A class for analyzing credit risk factors and patterns.
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize the CreditAnalyzer with a DataFrame.
        
        Args:
            data (pd.DataFrame): Input data containing credit information
        """
        self.data = data
    
    def calculate_risk_metrics(self) -> Dict:
        """
        Calculate key risk metrics from the data.
        
        Returns:
            Dict: Dictionary containing calculated risk metrics
        """
        metrics = {
            'default_rate': self.data['default'].mean(),
            'avg_credit_score': self.data['credit_score'].mean(),
            'high_risk_proportion': (self.data['credit_score'] < 600).mean()
        }
        return metrics
    
    def segment_analysis(self, segment_column: str) -> pd.DataFrame:
        """
        Perform analysis by customer segments.
        
        Args:
            segment_column (str): Column name to use for segmentation
        
        Returns:
            pd.DataFrame: Analysis results by segment
        """
        segment_stats = self.data.groupby(segment_column).agg({
            'default': 'mean',
            'credit_score': 'mean',
            'loan_amount': 'mean'
        }).round(2)
        
        return segment_stats
    
    def risk_correlation_analysis(self, features: List[str]) -> pd.DataFrame:
        """
        Analyze correlations between risk factors.
        
        Args:
            features (List[str]): List of feature columns to analyze
        
        Returns:
            pd.DataFrame: Correlation matrix
        """
        return self.data[features].corr()
