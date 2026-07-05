# src/ml_engineering/feature_store.py
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class FeatureStoreManager:
    """
    مدیریت ساده فیچرها بدون نیاز به Feast
    """
    def __init__(self, features_path: Path = Path("data/features")):
        self.features_path = features_path
        self.features_path.mkdir(parents=True, exist_ok=True)
        self.feature_cache = {}

    def generate_features(self, transactions_df: pd.DataFrame) -> pd.DataFrame:
        """تولید فیچرهای جدید از دیتای خام (همان کد قبلی)"""
        features = transactions_df.copy()
        
        # فیچرهای زمانی
        features['transaction_hour'] = pd.to_datetime(features['timestamp']).dt.hour
        features['transaction_day_of_week'] = pd.to_datetime(features['timestamp']).dt.dayofweek
        
        # فیچرهای آماری (با همان منطق قبلی)
        features['amount'] = features['amount'].astype(float)
        features['avg_transaction_7d'] = features.groupby('user_id')['amount'].transform(
            lambda x: x.rolling(window=7, min_periods=1).mean()
        )
        features['transaction_frequency_7d'] = features.groupby('user_id').cumcount()
        features['amount_to_avg_ratio'] = features['amount'] / (features['avg_transaction_7d'] + 1e-6)
        
        # فیچرهای تعاملی
        features['amount_hour_interaction'] = features['amount'] * features['transaction_hour'] / 100
        
        return features

    def save_features(self, features_df: pd.DataFrame, name: str = "current"):
        """ذخیره فیچرها در فایل"""
        filepath = self.features_path / f"{name}.parquet"
        features_df.to_parquet(filepath, index=False)
        logger.info(f"✅ Features saved to {filepath}")
        return filepath

    def load_features(self, name: str = "current") -> pd.DataFrame:
        """بارگذاری فیچرها از فایل"""
        filepath = self.features_path / f"{name}.parquet"
        if filepath.exists():
            df = pd.read_parquet(filepath)
            logger.info(f"✅ Features loaded from {filepath}")
            return df
        logger.warning(f"⚠️ Features file not found: {filepath}")
        return pd.DataFrame()