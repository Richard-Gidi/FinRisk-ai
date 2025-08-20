"""
Fraud analysis utilities for transaction monitoring and risk assessment.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

class FraudAnalyzer:
    """
    A class for analyzing transaction patterns and detecting potential fraud.
    """
    
    def __init__(self, data: Dict[str, pd.DataFrame]):
        """
        Initialize the FraudAnalyzer with transaction and customer data.
        
        Args:
            data (Dict[str, pd.DataFrame]): Dictionary containing:
                - transactions: Transaction data
                - customer_data: Customer profile data
        """
        self.transactions = data['transactions'].copy()
        self.customer_data = data.get('customer_data')
        
        # Add derived features
        self._add_time_features()
        self._add_amount_features()
    
    def _add_time_features(self):
        """Add time-based features to transactions."""
        self.transactions['hour'] = self.transactions['transaction_date'].dt.hour
        self.transactions['day_of_week'] = self.transactions['transaction_date'].dt.day_of_week
        self.transactions['is_weekend'] = self.transactions['day_of_week'].isin([5, 6]).astype(int)
        self.transactions['month'] = self.transactions['transaction_date'].dt.month
        self.transactions['day_of_month'] = self.transactions['transaction_date'].dt.day
    
    def _add_amount_features(self):
        """Add amount-based features to transactions."""
        # Calculate customer-level amount statistics
        amount_stats = self.transactions.groupby('customer_id')['amount'].agg(['mean', 'std']).fillna(0)
        amount_stats.columns = ['customer_avg_amount', 'customer_std_amount']
        
        # Merge back to transactions
        self.transactions = self.transactions.merge(
            amount_stats, 
            left_on='customer_id', 
            right_index=True, 
            how='left'
        )
        
        # Calculate z-scores for transaction amounts
        self.transactions['amount_zscore'] = (
            self.transactions['amount'] - self.transactions['customer_avg_amount']
        ) / self.transactions['customer_std_amount'].replace(0, 1)
    
    def calculate_velocity_metrics(self, 
                                 time_windows: List[str] = ['1H', '24H', '7D'],
                                 amount_threshold: Optional[float] = None) -> Dict[str, pd.DataFrame]:
        """
        Calculate transaction velocity metrics across multiple time windows.
        
        Args:
            time_windows (List[str]): Time windows for aggregation
            amount_threshold (float, optional): Threshold for high-value transactions
        
        Returns:
            Dict[str, pd.DataFrame]: Velocity metrics by time window
        """
        velocities = {}
        transactions = self.transactions[self.transactions['amount'] > amount_threshold] if amount_threshold else self.transactions
        
        for window in time_windows:
            velocity = transactions.groupby([
                'customer_id', 
                pd.Grouper(key='transaction_date', freq=window)
            ]).agg({
                'transaction_id': 'count',
                'amount': ['sum', 'mean', 'std'],
                'merchant_category': 'nunique',
                'location': 'nunique'
            })
            
            velocity.columns = [
                'tx_count', 'total_amount', 'avg_amount', 
                'std_amount', 'unique_merchants', 'unique_locations'
            ]
            velocities[window] = velocity.reset_index()
        
        return velocities
    
    def detect_anomalies(self, 
                        method: str = 'zscore',
                        features: List[str] = ['amount', 'tx_count'],
                        threshold: float = 3.0,
                        contamination: float = 0.1) -> pd.DataFrame:
        """
        Detect anomalous transactions using multiple methods.
        
        Args:
            method (str): Detection method ('zscore' or 'isolation_forest')
            features (List[str]): Features to use for anomaly detection
            threshold (float): Z-score threshold for anomaly detection
            contamination (float): Expected proportion of anomalies
        
        Returns:
            pd.DataFrame: Anomalous transactions with detection scores
        """
        if method == 'zscore':
            # Z-score based detection
            z_scores = {}
            for feature in features:
                z_scores[f'{feature}_zscore'] = np.abs(
                    (self.transactions[feature] - self.transactions[feature].mean()) 
                    / self.transactions[feature].std()
                )
            
            z_scores_df = pd.DataFrame(z_scores)
            anomalies = self.transactions[
                (z_scores_df > threshold).any(axis=1)
            ].copy()
            anomalies['anomaly_score'] = z_scores_df.max(axis=1)
            
        elif method == 'isolation_forest':
            # Isolation Forest based detection
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(self.transactions[features])
            
            clf = IsolationForest(
                contamination=contamination,
                random_state=42
            )
            
            scores = clf.fit_predict(features_scaled)
            anomalies = self.transactions[scores == -1].copy()
            anomalies['anomaly_score'] = clf.score_samples(features_scaled)[scores == -1]
        
        return anomalies
    
    def analyze_patterns(self) -> Dict[str, pd.DataFrame]:
        """
        Analyze various transaction patterns.
        
        Returns:
            Dict[str, pd.DataFrame]: Dictionary containing:
                - temporal_patterns: Time-based transaction patterns
                - merchant_patterns: Merchant category analysis
                - location_patterns: Geographic analysis
                - amount_patterns: Transaction amount analysis
        """
        patterns = {}
        
        # Temporal patterns
        patterns['temporal_patterns'] = {
            'hourly': self.transactions.groupby('hour').agg({
                'transaction_id': 'count',
                'amount': ['sum', 'mean'],
                'fraud_flag': ['count', 'mean']
            }),
            'daily': self.transactions.groupby('day_of_week').agg({
                'transaction_id': 'count',
                'amount': ['sum', 'mean'],
                'fraud_flag': ['count', 'mean']
            }),
            'monthly': self.transactions.groupby('month').agg({
                'transaction_id': 'count',
                'amount': ['sum', 'mean'],
                'fraud_flag': ['count', 'mean']
            })
        }
        
        # Merchant patterns
        patterns['merchant_patterns'] = self.transactions.groupby('merchant_category').agg({
            'transaction_id': 'count',
            'amount': ['mean', 'std'],
            'fraud_flag': ['count', 'mean'],
            'customer_id': 'nunique'
        })
        
        # Location patterns
        patterns['location_patterns'] = self.transactions.groupby('location').agg({
            'transaction_id': 'count',
            'amount': ['mean', 'std'],
            'fraud_flag': ['count', 'mean'],
            'customer_id': 'nunique'
        })
        
        # Amount patterns
        amount_bins = pd.qcut(self.transactions['amount'], q=10)
        patterns['amount_patterns'] = self.transactions.groupby(amount_bins).agg({
            'transaction_id': 'count',
            'fraud_flag': ['count', 'mean'],
            'customer_id': 'nunique'
        })
        
        return patterns
    
    def identify_high_risk_segments(self,
                                  min_transactions: int = 100,
                                  risk_threshold: float = 0.1) -> Dict[str, pd.DataFrame]:
        """
        Identify high-risk segments across different dimensions.
        
        Args:
            min_transactions (int): Minimum number of transactions to consider
            risk_threshold (float): Threshold for fraud rate to flag as high risk
        
        Returns:
            Dict[str, pd.DataFrame]: High-risk segments by different dimensions
        """
        risk_segments = {}
        
        # Filter base data
        def get_risk_segments(group_cols):
            segments = self.transactions.groupby(group_cols).agg({
                'transaction_id': 'count',
                'fraud_flag': ['count', 'mean'],
                'amount': ['sum', 'mean']
            }).reset_index()
            
            segments.columns = group_cols + [
                'transaction_count', 'fraud_count', 'fraud_rate',
                'total_amount', 'avg_amount'
            ]
            
            # Filter for high-risk segments
            high_risk = segments[
                (segments['transaction_count'] >= min_transactions) &
                (segments['fraud_rate'] >= risk_threshold)
            ].sort_values('fraud_rate', ascending=False)
            
            return high_risk
        
        # Get high-risk segments by different dimensions
        risk_segments['merchant'] = get_risk_segments(['merchant_category'])
        risk_segments['location'] = get_risk_segments(['location'])
        risk_segments['temporal'] = get_risk_segments(['hour', 'is_weekend'])
        
        # Get high-risk amount ranges
        amount_bins = pd.qcut(self.transactions['amount'], q=10, labels=False)
        self.transactions['amount_bin'] = amount_bins
        risk_segments['amount'] = get_risk_segments(['amount_bin'])
        
        return risk_segments
    
    def generate_risk_report(self) -> Dict[str, pd.DataFrame]:
        """
        Generate a comprehensive risk analysis report.
        
        Returns:
            Dict[str, pd.DataFrame]: Risk analysis report with multiple components
        """
        report = {}
        
        # Overall statistics
        report['summary'] = pd.DataFrame({
            'total_transactions': len(self.transactions),
            'total_fraud': self.transactions['fraud_flag'].sum(),
            'fraud_rate': self.transactions['fraud_flag'].mean(),
            'total_amount': self.transactions['amount'].sum(),
            'avg_transaction': self.transactions['amount'].mean(),
            'unique_customers': self.transactions['customer_id'].nunique(),
            'unique_merchants': self.transactions['merchant_category'].nunique(),
            'unique_locations': self.transactions['location'].nunique()
        }, index=[0])
        
        # Pattern analysis
        report['patterns'] = self.analyze_patterns()
        
        # Risk segments
        report['risk_segments'] = self.identify_high_risk_segments()
        
        # Velocity metrics
        report['velocity'] = self.calculate_velocity_metrics()
        
        # Anomaly detection
        report['anomalies'] = {
            'zscore': self.detect_anomalies(method='zscore'),
            'isolation_forest': self.detect_anomalies(method='isolation_forest')
        }
        
        return report
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
