# scripts/test_model.py
import sys
from pathlib import Path
import pandas as pd
import joblib

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ml_engineering.inference import FraudInferenceService

def test_model():
    print("=" * 50)
    print("🧪 Testing Fraud Detection Model")
    print("=" * 50)
    
    # 1. بارگذاری مدل
    inference = FraudInferenceService()
    
    if inference.model is None:
        print("❌ Model not loaded!")
        return
    
    print("✅ Model loaded successfully!")
    
    # 2. ایجاد یک تراکنش نمونه
    sample_transaction = {
        'transaction_id': 'TEST001',
        'user_id': 'user_123',
        'amount': 1500.0,
        'timestamp': '2024-07-01T10:30:00',
        'merchant': 'online_retail',
        'location': 'Tehran',
        'transaction_type': 'online',
        'device_id': 'DEV001',
        'ip_address': '192.168.1.1',
        'features': {
            'transaction_hour': 10,
            'transaction_day_of_week': 3,
            'avg_transaction_7d': 1200.0,
            'transaction_frequency_7d': 5,
        }
    }
    
    print("\n📝 Sample Transaction:")
    print(f"   ID: {sample_transaction['transaction_id']}")
    print(f"   Amount: ${sample_transaction['amount']}")
    print(f"   User: {sample_transaction['user_id']}")
    
    # 3. پیش‌بینی
    print("\n🔮 Making prediction...")
    result = inference.predict(sample_transaction)
    
    print("\n📊 Prediction Result:")
    print(f"   Fraud Probability: {result.get('fraud_probability', 0) * 100:.2f}%")
    print(f"   Is Fraud: {'🚨 YES' if result.get('is_fraud') else '✅ NO'}")
    print(f"   Threshold: {result.get('threshold', 0.5)}")
    
    if 'feature_importance' in result:
        print("\n📈 Top Features:")
        importance = result['feature_importance']
        sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
        for feat, imp in sorted_imp:
            print(f"   {feat}: {imp:.4f}")

if __name__ == "__main__":
    test_model()