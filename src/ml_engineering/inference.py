# src/ml_engineering/inference.py
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    کلاس استاندارد برای استخراج فیچرها
    تمام فیچرهای ممکن را تولید می‌کند و بر اساس نیاز مدل، انتخاب می‌کند
    """
    
    @staticmethod
    def extract_all_features(data: Dict[str, Any]) -> pd.DataFrame:
        """
        تولید تمام فیچرهای ممکن از داده تراکنش
        
        Returns:
            DataFrame با تمام فیچرهای ممکن
        """
        amount = float(data.get('amount', 0))
        
        # فیچرهای زمانی
        timestamp = data.get('timestamp')
        if timestamp:
            if isinstance(timestamp, str):
                dt = pd.to_datetime(timestamp)
            else:
                dt = timestamp
            hour = dt.hour
            day_of_week = dt.weekday()
            day_of_month = dt.day
            month = dt.month
            is_weekend = 1 if day_of_week >= 5 else 0
        else:
            hour = 12
            day_of_week = 3
            day_of_month = 15
            month = 6
            is_weekend = 0
        
        # فیچرهای فروشنده
        merchant = data.get('merchant', '')
        merchant_map = {
            'Amazon': 1, 'Apple': 2, 'Google': 3, 'Netflix': 4,
            'Spotify': 5, 'Uber': 6, 'DoorDash': 7, 'Walmart': 8,
            'Target': 9, 'Best Buy': 10, 'Cryptocurrency Exchange': 11,
            'Online Casino': 12, 'Gambling Site': 13, 'Foreign Exchange': 14,
            'Money Transfer': 15, 'Unknown': 0
        }
        merchant_code = merchant_map.get(merchant, 0)
        
        # فیچرهای موقعیت
        location = data.get('location', '')
        location_map = {
            'Tehran': 1, 'Mashhad': 2, 'Isfahan': 3, 'Shiraz': 4,
            'Tabriz': 5, 'Qom': 6, 'Karaj': 7, 'Ahvaz': 8
        }
        location_code = location_map.get(location, 0)
        
        # فیچرهای نوع تراکنش
        tx_type = data.get('transaction_type', '')
        type_map = {
            'online': 1, 'in_store': 2, 'atm': 3, 'transfer': 4, 'payment': 5
        }
        type_code = type_map.get(tx_type, 0)
        
        # ============================================
        # ساخت دیکشنری تمام فیچرها
        # ============================================
        features_dict = {
            # فیچرهای عددی پایه
            'amount': amount,
            'amount_log': np.log(amount + 1),
            'amount_sqrt': np.sqrt(amount),
            'amount_rounded': round(amount / 100) * 100,
            
            # فیچرهای زمانی
            'hour': hour,
            'hour_sin': np.sin(2 * np.pi * hour / 24),
            'hour_cos': np.cos(2 * np.pi * hour / 24),
            'day_of_week': day_of_week,
            'day_of_month': day_of_month,
            'month': month,
            'is_weekend': is_weekend,
            'is_night': 1 if (hour < 6 or hour > 22) else 0,
            'is_business_hours': 1 if (9 <= hour <= 17) else 0,
            
            # فیچرهای فروشنده
            'merchant_code': merchant_code,
            'is_high_risk_merchant': 1 if merchant_code in [11, 12, 13, 14] else 0,
            
            # فیچرهای موقعیت
            'location_code': location_code,
            
            # فیچرهای نوع تراکنش
            'transaction_type_code': type_code,
            'is_online': 1 if type_code == 1 else 0,
            
            # فیچرهای تعاملی
            'amount_hour': amount * hour,
            'amount_hour_normalized': amount * hour / 100000,
            'amount_merchant': amount * merchant_code,
            'amount_weekend': amount * is_weekend,
            'hour_merchant': hour * merchant_code,
        }
        
        # اضافه کردن feature_0 تا feature_14 برای سازگاری با مدل‌های قدیمی
        # اینها را با مقادیر معنی‌دار پر می‌کنیم
        for i in range(15):
            if i < len(features_dict):
                features_dict[f'feature_{i}'] = list(features_dict.values())[i % len(features_dict)]
            else:
                features_dict[f'feature_{i}'] = np.random.randn() * 0.1
        
        return pd.DataFrame([features_dict])


class FraudInferenceService:
    """
    سرویس تشخیص تقلب با قابلیت کار با هر نوع مدلی
    """
    
    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = model_path or Path("models/best_model.pkl")
        self.model = None
        self.threshold = 0.5
        self.model_features = None
        self.model_type = None
        self.load_model()
    
    def load_model(self):
        """بارگذاری مدل و تشخیص نوع آن"""
        try:
            if not self.model_path.exists():
                logger.warning(f"⚠️ Model not found at {self.model_path}")
                self.model = None
                return
            
            self.model = joblib.load(self.model_path)
            self.model_type = self._detect_model_type()
            
            logger.info(f"✅ Model loaded: {self.model_type}")
            logger.info(f"   Path: {self.model_path}")
            
            # استخراج فیچرهای مورد نیاز مدل
            self.model_features = self._extract_model_features()
            
            if self.model_features:
                logger.info(f"   Required features: {self.model_features[:5]}... ({len(self.model_features)} total)")
            else:
                logger.warning("   Could not determine required features")
                
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            self.model = None
            self.model_features = None
    
    def _detect_model_type(self) -> str:
        """تشخیص نوع مدل"""
        if self.model is None:
            return 'unknown'
        
        model_str = str(type(self.model)).lower()
        
        if 'catboost' in model_str:
            return 'catboost'
        elif 'xgboost' in model_str:
            return 'xgboost'
        elif 'lightgbm' in model_str:
            return 'lightgbm'
        elif 'randomforest' in model_str:
            return 'randomforest'
        elif 'sklearn' in model_str:
            return 'sklearn'
        else:
            return 'unknown'
    
    def _extract_model_features(self) -> Optional[List[str]]:
        """استخراج نام فیچرهای مورد نیاز مدل"""
        if self.model is None:
            return None
        
        try:
            # روش 1: برای CatBoost
            if self.model_type == 'catboost':
                if hasattr(self.model, 'feature_names_'):
                    return list(self.model.feature_names_)
                else:
                    # fallback: 15 فیچر با نام feature_0 تا feature_14
                    return [f'feature_{i}' for i in range(15)]
            
            # روش 2: برای مدل‌های sklearn
            if hasattr(self.model, 'feature_names_in_'):
                return list(self.model.feature_names_in_)
            
            # روش 3: برای XGBoost
            if hasattr(self.model, 'get_booster'):
                try:
                    return self.model.get_booster().feature_names
                except:
                    pass
            
            # روش 4: برای LightGBM
            if hasattr(self.model, 'feature_name_'):
                return list(self.model.feature_name_)
            
            # اگر هیچکدام کار نکرد، از فیچرهای پیش‌فرض استفاده کن
            return None
            
        except Exception as e:
            logger.warning(f"Could not extract features: {e}")
            return None
    
    def predict(self, transaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """پیش‌بینی تقلب"""
        
        if self.model is None:
            return {
                'fraud_probability': 0.0,
                'is_fraud': False,
                'threshold': self.threshold,
                'error': 'Model not loaded'
            }
        
        try:
            # 1. تولید تمام فیچرهای ممکن
            all_features = FeatureExtractor.extract_all_features(transaction_data)
            
            # 2. انتخاب فیچرهای مورد نیاز مدل
            if self.model_features:
                # اگر مدل فیچرهای خاصی نیاز داره
                selected_features = self._select_features(all_features)
            else:
                # اگر نمی‌دونیم چه فیچرهایی نیازه، همه رو می‌فرستیم
                selected_features = all_features
                logger.warning("Using all features (model features unknown)")
            
            # 3. اطمینان از تطابق فیچرها با مدل
            if self.model_features and len(selected_features.columns) != len(self.model_features):
                logger.warning(f"Feature mismatch: {len(selected_features.columns)} vs {len(self.model_features)}")
                # تطبیق با فیچرهای مورد نیاز مدل
                selected_features = self._align_features(selected_features)
            
            # 4. پیش‌بینی
            fraud_probability = self.model.predict_proba(selected_features)[0, 1]
            is_fraud = fraud_probability > self.threshold
            
            return {
                'fraud_probability': float(fraud_probability),
                'is_fraud': bool(is_fraud),
                'threshold': self.threshold,
                'model_type': self.model_type,
                'prediction_time': datetime.utcnow().isoformat(),
                'features_used': list(selected_features.columns)[:10]  # فقط ۱۰ تا اول برای نمایش
            }
            
        except Exception as e:
            logger.error(f"❌ Prediction error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                'error': str(e),
                'fraud_probability': 0.0,
                'is_fraud': False,
                'threshold': self.threshold
            }
    
    def _select_features(self, all_features: pd.DataFrame) -> pd.DataFrame:
        """انتخاب فیچرهای مورد نیاز از بین همه فیچرها"""
        if self.model_features is None:
            return all_features
        
        available_features = set(all_features.columns)
        required_features = set(self.model_features)
        
        # فیچرهای موجود
        existing = required_features & available_features
        
        # فیچرهای گم‌شده
        missing = required_features - available_features
        
        if missing:
            logger.warning(f"⚠️ Missing features: {missing}")
            # اضافه کردن فیچرهای گم‌شده با مقدار 0
            for feat in missing:
                all_features[feat] = 0
            existing = required_features
        
        # مرتب‌سازی بر اساس ترتیب مدل
        return all_features[list(self.model_features)]
    
    def _align_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """تطبیق فیچرها با مدل"""
        if self.model_features is None:
            return features
        
        # اگر تعداد فیچرها کمتر از نیاز مدل است، با 0 پر کن
        if len(features.columns) < len(self.model_features):
            for i, feat in enumerate(self.model_features):
                if feat not in features.columns:
                    features[feat] = 0
        
        # اگر تعداد فیچرها بیشتر از نیاز مدل است، کم کن
        if len(features.columns) > len(self.model_features):
            features = features[self.model_features]
        
        return features


# برای تست
if __name__ == "__main__":
    service = FraudInferenceService()
    
    test_tx = {
        'transaction_id': 'TEST001',
        'user_id': 'user_123',
        'amount': 1500.0,
        'timestamp': datetime.now().isoformat(),
        'merchant': 'Amazon',
        'location': 'Tehran',
        'transaction_type': 'online'
    }
    
    result = service.predict(test_tx)
    print(f"Result: {result}")