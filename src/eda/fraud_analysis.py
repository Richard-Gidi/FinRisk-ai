"""
Fraud analysis utilities for transaction monitoring.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime, timedelta

class FraudAnalyzer:
    """
    A class for analyzing transaction patterns and detecting potential fraud.
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize the FraudAnalyzer with transaction data.
        
        Args:
            data (pd.DataFrame): Transaction data
        """
        self.data = data
    
    def calculate_velocity_metrics(self, 
                                 time_window: str = '1H',
                                 amount_threshold: float = None) -> pd.DataFrame:
        """
        Calculate transaction velocity metrics.
        
        Args:
            time_window (str): Time window for aggregation (default: '1H')
            amount_threshold (float): Optional threshold for high-value transactions
        
        Returns:
            pd.DataFrame: Velocity metrics by customer
        """
        if amount_threshold:
            transactions = self.data[self.data['amount'] > amount_threshold]
        else:
            transactions = self.data
            
        velocity = transactions.groupby(
            ['customer_id', pd.Grouper(key='transaction_date', freq=time_window)]
        ).agg({
            'transaction_id': 'count',
            'amount': ['sum', 'mean']
        })
        
        return velocity
    
    def detect_anomalies(self, 
                        features: List[str],
                        threshold: float = 3.0) -> pd.DataFrame:
        """
        Detect anomalous transactions using z-score method.
        
        Args:
            features (List[str]): Features to use for anomaly detection
            threshold (float): Z-score threshold for anomaly detection
        
        Returns:
            pd.DataFrame: Anomalous transactions
        """
        z_scores = {}
        for feature in features:
            z_scores[f'{feature}_zscore'] = np.abs(
                (self.data[feature] - self.data[feature].mean()) 
                / self.data[feature].std()
            )
        
        z_scores_df = pd.DataFrame(z_scores)
        anomalies = self.data[
            (z_scores_df > threshold).any(axis=1)
        ].copy()
        
        return anomalies
    
    def analyze_geographic_patterns(self) -> pd.DataFrame:
        """
        Analyze transaction patterns by geography.
        
        Returns:
            pd.DataFrame: Geographic risk analysis
        """
        geo_analysis = self.data.groupby('merchant_location').agg({
            'transaction_id': 'count',
            'amount': ['mean', 'std'],
            'is_fraud': 'mean'
        })
        
        return geo_analysis.round(3)
    
    def get_high_risk_merchants(self, 
                              min_transactions: int = 100,
                              risk_threshold: float = 0.1) -> pd.DataFrame:
        """
        Identify high-risk merchant categories.
        
        Args:
            min_transactions (int): Minimum number of transactions to consider
            risk_threshold (float): Fraud rate threshold for high risk
        
        Returns:
            pd.DataFrame: High-risk merchant categories
        """
        merchant_risk = self.data.groupby('merchant_category').agg({
            'transaction_id': 'count',
            'is_fraud': 'mean'
        })
        
        high_risk = merchant_risk[
            (merchant_risk['transaction_id'] >= min_transactions) &
            (merchant_risk['is_fraud'] >= risk_threshold)
        ]
        
        return high_risk.sort_values('is_fraud', ascending=False)
