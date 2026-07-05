"""
تست‌های واحد برای API
"""

import pytest
from fastapi.testclient import TestClient
from datetime import datetime, timedelta
import json
import sys
from pathlib import Path

# افزودن مسیر src به PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api.main import app
from src.core.database import db_manager, Transaction
from src.ml_engineering.inference import FraudInferenceService

# ============== Client ==============

client = TestClient(app)

# ============== Fixtures ==============

@pytest.fixture
def sample_transaction():
    """نمونه تراکنش برای تست"""
    return {
        "transaction_id": "TEST001",
        "user_id": "test_user",
        "amount": 250.75,
        "timestamp": datetime.utcnow().isoformat(),
        "merchant": "Amazon",
        "location": "Tehran",
        "transaction_type": "online",
        "device_id": "TEST-DEV-001",
        "ip_address": "192.168.1.100"
    }

@pytest.fixture
def sample_transactions():
    """لیست نمونه تراکنش‌ها برای تست"""
    transactions = []
    for i in range(5):
        trans = {
            "transaction_id": f"TEST{i:03d}",
            "user_id": f"test_user_{i}",
            "amount": round(100 + i * 50.5, 2),
            "timestamp": (datetime.utcnow() - timedelta(hours=i)).isoformat(),
            "merchant": ["Amazon", "Google", "Apple", "Netflix", "Spotify"][i % 5],
            "location": "Tehran",
            "transaction_type": "online",
            "device_id": f"TEST-DEV-{i:03d}",
            "ip_address": f"192.168.1.{100 + i}"
        }
        transactions.append(trans)
    return transactions

@pytest.fixture
def setup_test_db():
    """تنظیم دیتابیس تست"""
    db_manager.create_tables()
    
    # پاک کردن دیتای قبلی
    session = db_manager.get_session()
    try:
        session.query(Transaction).delete()
        session.commit()
    finally:
        session.close()
    
    yield
    
    # پاک کردن بعد از تست
    session = db_manager.get_session()
    try:
        session.query(Transaction).delete()
        session.commit()
    finally:
        session.close()

# ============== تست‌های سلامتی ==============

def test_health_check():
    """تست اندپوینت سلامت"""
    response = client.get("/health")
    assert response.status_code == 200
    
    data = response.json()
    assert data["status"] == "healthy"
    assert "components" in data
    assert "database" in data["components"]
    assert "models" in data["components"]

def test_root_endpoint():
    """تست صفحه اصلی"""
    response = client.get("/")
    assert response.status_code == 200
    
    data = response.json()
    assert data["name"] == "Fraud Detection Platform API"
    assert data["version"] == "1.0.0"
    assert "documentation" in data
    assert "health" in data

def test_info_endpoint():
    """تست اندپوینت اطلاعات"""
    response = client.get("/info")
    assert response.status_code == 200
    
    data = response.json()
    assert "application" in data
    assert "api" in data
    assert "database" in data
    assert "ml" in data

# ============== تست‌های تشخیص تقلب ==============

def test_detect_fraud_success(sample_transaction, setup_test_db):
    """تست موفق تشخیص تقلب"""
    response = client.post("/api/v1/fraud/detect", json=sample_transaction)
    assert response.status_code == 200
    
    data = response.json()
    assert data["status"] == "success"
    assert "data" in data
    assert "transaction_id" in data["data"]
    assert data["data"]["transaction_id"] == sample_transaction["transaction_id"]
    assert "prediction" in data["data"]

def test_detect_fraud_invalid_amount():
    """تست تشخیص تقلب با مبلغ نامعتبر"""
    transaction = {
        "transaction_id": "TEST002",
        "user_id": "test_user",
        "amount": -100,  # مبلغ منفی
        "timestamp": datetime.utcnow().isoformat(),
        "merchant": "Amazon"
    }
    
    response = client.post("/api/v1/fraud/detect", json=transaction)
    assert response.status_code == 422  # Validation error

def test_detect_fraud_missing_fields():
    """تست تشخیص تقلب با فیلدهای ناقص"""
    transaction = {
        "transaction_id": "TEST003",
        "user_id": "test_user"
        # missing amount and timestamp
    }
    
    response = client.post("/api/v1/fraud/detect", json=transaction)
    assert response.status_code == 422  # Validation error

def test_batch_detect_fraud(sample_transactions, setup_test_db):
    """تست دسته‌ای تشخیص تقلب"""
    response = client.post("/api/v1/fraud/batch-detect", json=sample_transactions)
    assert response.status_code == 200
    
    data = response.json()
    assert data["status"] == "success"
    assert "data" in data
    assert len(data["data"]) == len(sample_transactions)
    
    # بررسی هر نتیجه
    for result in data["data"]:
        assert "transaction_id" in result
        assert "prediction" in result

def test_batch_detect_fraud_limit():
    """تست محدودیت دسته‌ای تشخیص تقلب"""
    # تولید 150 تراکنش (بیش از حد مجاز 100)
    transactions = []
    for i in range(150):
        transactions.append({
            "transaction_id": f"TEST{i:03d}",
            "user_id": "test_user",
            "amount": 100.0,
            "timestamp": datetime.utcnow().isoformat(),
            "merchant": "Amazon"
        })
    
    response = client.post("/api/v1/fraud/batch-detect", json=transactions)
    assert response.status_code == 422  # Validation error

# ============== تست‌های تاریخچه ==============

def test_get_user_history(sample_transaction, setup_test_db):
    """تست دریافت تاریخچه کاربر"""
    # ابتدا یک تراکنش ثبت کن
    client.post("/api/v1/fraud/detect", json=sample_transaction)
    
    # سپس تاریخچه را بگیر
    response = client.get(f"/api/v1/fraud/history/{sample_transaction['user_id']}")
    assert response.status_code == 200
    
    data = response.json()
    assert data["status"] == "success"
    assert "data" in data
    assert data["user_id"] == sample_transaction["user_id"]
    assert len(data["data"]) >= 1

def test_get_user_history_invalid_user():
    """تست دریافت تاریخچه کاربر ناموجود"""
    response = client.get("/api/v1/fraud/history/non_existent_user")
    assert response.status_code == 200  # باز هم موفق اما لیست خالی
    data = response.json()
    assert len(data["data"]) == 0

def test_get_user_history_with_days():
    """تست دریافت تاریخچه با تعداد روزهای مشخص"""
    response = client.get("/api/v1/fraud/history/test_user?days=7")
    assert response.status_code == 200
    
    data = response.json()
    assert data["days"] == 7

# ============== تست‌های آمار ==============

def test_get_fraud_stats(setup_test_db):
    """تست دریافت آمار تقلب"""
    # ثبت چند تراکنش
    for i in range(10):
        transaction = {
            "transaction_id": f"STATS{i:03d}",
            "user_id": f"user_{i}",
            "amount": 100.0 + i * 10,
            "timestamp": (datetime.utcnow() - timedelta(hours=i)).isoformat(),
            "merchant": "Amazon"
        }
        client.post("/api/v1/fraud/detect", json=transaction)
    
    response = client.get("/api/v1/fraud/stats")
    assert response.status_code == 200
    
    data = response.json()
    assert data["status"] == "success"
    assert "data" in data
    
    stats = data["data"]
    assert "total_transactions" in stats
    assert "fraud_transactions" in stats
    assert "fraud_rate" in stats
    assert "avg_amount" in stats
    assert "period_days" in stats

def test_get_fraud_stats_with_days():
    """تست دریافت آمار با تعداد روزهای مشخص"""
    response = client.get("/api/v1/fraud/stats?days=3")
    assert response.status_code == 200
    
    data = response.json()
    assert data["data"]["period_days"] == 3

# ============== تست‌های آستانه ==============

def test_update_threshold():
    """تست به‌روزرسانی آستانه"""
    new_threshold = 0.7
    response = client.post(f"/api/v1/fraud/update-threshold?threshold={new_threshold}")
    assert response.status_code == 200
    
    data = response.json()
    assert data["status"] == "success"
    assert data["new_threshold"] == new_threshold
    assert "old_threshold" in data

def test_update_threshold_invalid():
    """تست به‌روزرسانی آستانه نامعتبر"""
    # آستانه بیش از 1
    response = client.post("/api/v1/fraud/update-threshold?threshold=1.5")
    assert response.status_code == 422
    
    # آستانه کمتر از 0
    response = client.post("/api/v1/fraud/update-threshold?threshold=-0.5")
    assert response.status_code == 422

# ============== تست‌های عملکرد مدل ==============

def test_get_model_performance(setup_test_db):
    """تست دریافت عملکرد مدل"""
    # ثبت چند تراکنش برای داشتن پیش‌بینی
    for i in range(5):
        transaction = {
            "transaction_id": f"PERF{i:03d}",
            "user_id": f"user_{i}",
            "amount": 100.0 + i * 50,
            "timestamp": datetime.utcnow().isoformat(),
            "merchant": "Amazon"
        }
        client.post("/api/v1/fraud/detect", json=transaction)
    
    response = client.get("/api/v1/fraud/performance")
    assert response.status_code == 200
    
    data = response.json()
    assert data["status"] == "success"
    assert "data" in data
    
    perf = data["data"]
    assert "total_predictions" in perf
    assert "fraud_predicted" in perf

# ============== تست‌های پروفایل ریسک ==============

def test_get_user_risk_profile(sample_transaction, setup_test_db):
    """تست دریافت پروفایل ریسک کاربر"""
    # ثبت چند تراکنش برای کاربر
    for i in range(3):
        transaction = sample_transaction.copy()
        transaction["transaction_id"] = f"RISK{i:03d}"
        transaction["amount"] = 100.0 + i * 100
        client.post("/api/v1/fraud/detect", json=transaction)
    
    response = client.get(f"/api/v1/fraud/risk-profile/{sample_transaction['user_id']}")
    assert response.status_code == 200
    
    data = response.json()
    assert data["status"] == "success"
    assert "data" in data
    
    profile = data["data"]
    assert "user_id" in profile
    assert "risk_score" in profile
    assert "risk_level" in profile
    assert profile["risk_level"] in ["LOW", "MEDIUM", "HIGH"]

def test_get_user_risk_profile_no_transactions():
    """تست پروفایل ریسک کاربر بدون تراکنش"""
    response = client.get("/api/v1/fraud/risk-profile/non_existent_user")
    assert response.status_code == 200
    
    data = response.json()
    assert data["status"] == "success"
    assert data["data"]["message"] == "No transactions found for this user"

# ============== تست‌های خطا ==============

def test_invalid_route():
    """تست مسیر نامعتبر"""
    response = client.get("/invalid_route")
    assert response.status_code == 404

def test_invalid_method():
    """تست متد نامعتبر"""
    response = client.put("/api/v1/fraud/detect")
    assert response.status_code == 405  # Method not allowed

# ============== تست‌های استرس (اختیاری) ==============

@pytest.mark.slow
def test_stress_test():
    """تست استرس با درخواست‌های متوالی"""
    import time
    
    start_time = time.time()
    success_count = 0
    
    for i in range(100):
        transaction = {
            "transaction_id": f"STRESS{i:03d}",
            "user_id": f"stress_user",
            "amount": 100.0 + i,
            "timestamp": datetime.utcnow().isoformat(),
            "merchant": "Amazon"
        }
        response = client.post("/api/v1/fraud/detect", json=transaction)
        
        if response.status_code == 200:
            success_count += 1
    
    elapsed_time = time.time() - start_time
    
    print(f"\n📊 Stress Test Results:")
    print(f"   Total Requests: 100")
    print(f"   Success Rate: {success_count}%")
    print(f"   Time: {elapsed_time:.2f}s")
    print(f"   RPS: {100/elapsed_time:.2f}")
    
    assert success_count >= 90  # حداقل 90% موفقیت

# ============== اجرای تست‌ها ==============

if __name__ == "__main__":
    # اجرای تست‌ها با pytest
    pytest.main([__file__, "-v", "--tb=short"])