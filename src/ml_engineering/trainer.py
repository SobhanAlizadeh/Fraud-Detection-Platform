# src/ml_engineering/inference.py
"""
سرویس استنتاج با بارگذاری خودکار بهترین مدل از MLflow
"""

import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Any, Optional
from datetime import datetime
import mlflow
from mlflow.tracking import MlflowClient

from .feature_store import FeatureStoreManager
from ..core.database import db_manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FraudInferenceService:
    """
    سرویس استنتاج حرفه‌ای با بارگذاری خودکار مدل
    """
    
    def __init__(self, 
                 model_path: Optional[Path] = None,
                 mlflow_uri: str = "http://localhost:5000",
                 auto_update: bool = True):
        
        self.model_path = model_path or Path("models/best_model.pkl")
        self.feature_store = FeatureStoreManager()
        self.mlflow_uri = mlflow_uri
        self.auto_update = auto_update
        self.model = None
        self.threshold = 0.5
        self.model_version = None
        self.last_update_time = None
        
        # تنظیم MLflow
        mlflow.set_tracking_uri(mlflow_uri)
        self.client = MlflowClient()
        
        # بارگذاری مدل
        self._load_model()
        
        logger.info(f"✅ FraudInferenceService initialized with MLflow URI: {mlflow_uri}")
    
    def _load_model(self):
        """بارگذاری مدل از MLflow Registry یا فایل محلی"""
        
        # 1. تلاش برای بارگذاری از MLflow Registry
        try:
            model_name = "fraud_best_model"
            try:
                latest = self.client.get_latest_versions(model_name, stages=["Production"])
                if latest:
                    model_uri = f"models:/{model_name}/Production"
                    self.model = mlflow.sklearn.load_model(model_uri)
                    self.model_version = latest[0].version
                    self.last_update_time = datetime.now()
                    logger.info(f"✅ Model loaded from MLflow Registry: {model_name} (v{self.model_version})")
                    return
            except Exception as e:
                logger.warning(f"⚠️ Could not load from MLflow Registry: {e}")
        except Exception as e:
            logger.warning(f"⚠️ MLflow error: {e}")
        
        # 2. بارگذاری از فایل محلی
        if self.model_path.exists():
            try:
                self.model = joblib.load(self.model_path)
                self.model_version = "local"
                self.last_update_time = datetime.now()
                logger.info(f"✅ Model loaded from local file: {self.model_path}")
                return
            except Exception as e:
                logger.error(f"❌ Error loading local model: {e}")
        
        # 3. اگر مدلی وجود نداشت، یک مدل پیش‌فرض ایجاد کن
        logger.warning("⚠️ No model found. Creating default model...")
        self._create_default_model()
    
    def _create_default_model(self):
        """ایجاد مدل پیش‌فرض"""
        from sklearn.ensemble import RandomForestClassifier
        
        # داده‌های نمونه
        X = np.random.rand(200, 5)
        y = np.random.randint(0, 2, 200)
        
        # آموزش مدل
        self.model = RandomForestClassifier(n_estimators=10, random_state=42)
        self.model.fit(X, y)
        self.model_version = "default"
        
        # ذخیره
        self.model_path.parent.mkdir(exist_ok=True)
        joblib.dump(self.model, self.model_path)
        logger.info(f"✅ Default model created and saved to {self.model_path}")
    
    def check_and_update_model(self):
        """بررسی و به‌روزرسانی مدل از MLflow"""
        if not self.auto_update:
            return
        
        try:
            latest = self.client.get_latest_versions("fraud_best_model", stages=["Production"])
            if latest:
                latest_version = latest[0].version
                if self.model_version != latest_version:
                    logger.info(f"🔄 New model version available: {latest_version}")
                    model_uri = f"models:/fraud_best_model/Production"
                    self.model = mlflow.sklearn.load_model(model_uri)
                    self.model_version = latest_version
                    self.last_update_time = datetime.now()
                    logger.info(f"✅ Model updated to version {latest_version}")
        except Exception as e:
            logger.warning(f"⚠️ Could not check for model updates: {e}")
    
    def predict(self, transaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """پیش‌بینی تقلب"""
        
        try:
            # بررسی به‌روزرسانی مدل
            self.check_and_update_model()
            
            # تبدیل به DataFrame
            df = pd.DataFrame([transaction_data])
            
            # استخراج فیچرها
            features = self.feature_store.generate_features(df)
            
            # حذف ستون‌های غیرضروری
            feature_cols = [col for col in features.columns if col not in [
                'transaction_id', 'user_id', 'timestamp', 'is_fraud', 'id'
            ]]
            
            X = features[feature_cols]
            
            # پیش‌بینی
            if self.model is not None:
                fraud_probability = self.model.predict_proba(X)[0, 1]
                is_fraud = fraud_probability > self.threshold
                
                # اهمیت فیچرها
                feature_importance = None
                if hasattr(self.model, 'feature_importances_'):
                    feature_importance = dict(zip(
                        feature_cols,
                        self.model.feature_importances_.tolist()
                    ))
                
                # ذخیره نتیجه
                self._save_prediction(
                    transaction_id=transaction_data.get('transaction_id'),
                    fraud_probability=fraud_probability,
                    is_fraud_predicted=is_fraud,
                    features_used=feature_cols,
                    feature_importance=feature_importance
                )
                
                return {
                    'transaction_id': transaction_data.get('transaction_id'),
                    'fraud_probability': float(fraud_probability),
                    'is_fraud': bool(is_fraud),
                    'threshold': self.threshold,
                    'features_used': feature_cols,
                    'feature_importance': feature_importance,
                    'model_version': self.model_version or 'unknown',
                    'prediction_time': datetime.utcnow().isoformat()
                }
            else:
                logger.error("Model not loaded")
                return {'error': 'Model not available'}
                
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {'error': str(e)}
    
    def _save_prediction(self, transaction_id: str, fraud_probability: float,
                        is_fraud_predicted: bool, features_used: list,
                        feature_importance: Optional[dict] = None):
        """ذخیره نتیجه در دیتابیس"""
        try:
            session = db_manager.get_session()
            from ..core.database import Prediction
            
            prediction = Prediction(
                transaction_id=transaction_id,
                fraud_probability=fraud_probability,
                is_fraud_predicted=is_fraud_predicted,
                model_version=self.model_version or 'unknown',
                features_used=features_used,
                feature_importance=feature_importance
            )
            
            session.add(prediction)
            session.commit()
            session.close()
            
        except Exception as e:
            logger.error(f"Error saving prediction: {e}")
    
    def refresh_model(self):
        """بازخوانی اجباری مدل"""
        self._load_model()
        logger.info("✅ Model refreshed manually")
        return True
    
    def get_model_info(self) -> Dict[str, Any]:
        """دریافت اطلاعات مدل فعلی"""
        return {
            'model_version': self.model_version,
            'last_update_time': self.last_update_time.isoformat() if self.last_update_time else None,
            'threshold': self.threshold,
            'auto_update': self.auto_update,
            'model_loaded': self.model is not None,
            'mlflow_uri': self.mlflow_uri
        }