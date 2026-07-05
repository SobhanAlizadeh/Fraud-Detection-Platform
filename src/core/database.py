# src/core/database.py
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Boolean, BigInteger, Text, text
from sqlalchemy.dialects.postgresql import JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os
import logging

logger = logging.getLogger(__name__)

Base = declarative_base()

class Transaction(Base):
    __tablename__ = "transactions"
    
    transaction_id = Column(Text, primary_key=True)
    customer_id = Column(BigInteger, nullable=False)
    user_id = Column(String(50), nullable=True)
    merchant_id = Column(BigInteger, nullable=False)
    merchant = Column(String(100), nullable=True)
    amount = Column(Float, nullable=False)
    transaction_type = Column(Text, nullable=False)
    timestamp = Column(DateTime, nullable=False)
    is_fraud = Column(Integer, nullable=False)
    location = Column(String(100), nullable=True)
    device_id = Column(String(50), nullable=True)
    ip_address = Column(String(45), nullable=True)
    fraud_score = Column(Float, nullable=True)

class Prediction(Base):
    __tablename__ = "predictions"
    
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    transaction_id = Column(Text, nullable=False)
    fraud_probability = Column(Float, nullable=False)
    is_fraud_predicted = Column(Boolean, nullable=False)
    model_version = Column(String(20))
    prediction_time = Column(DateTime, default=datetime.utcnow)
    features_used = Column(JSON, nullable=True)
    feature_importance = Column(JSON, nullable=True)

class Customer(Base):
    __tablename__ = "customers"
    
    customer_id = Column(Integer, primary_key=True)
    name = Column(String(100))
    age = Column(Integer)
    job = Column(String(100))
    city = Column(String(100))
    account_age_days = Column(Integer)

class Merchant(Base):
    __tablename__ = "merchants"
    
    merchant_id = Column(Integer, primary_key=True)
    merchant_name = Column(String(100))
    category = Column(String(50))
    risk_level = Column(Integer)

class DatabaseManager:
    def __init__(self):
        DB_HOST = os.getenv("DB_HOST", "localhost")
        DB_PORT = os.getenv("DB_PORT", "5432")
        DB_NAME = os.getenv("DB_NAME", "fraud_db")
        DB_USER = os.getenv("DB_USER", "fraud_user")
        DB_PASSWORD = os.getenv("DB_PASSWORD", "fraud_pass")
        
        self.database_url = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        self.engine = create_engine(
            self.database_url,
            pool_pre_ping=True,  # بررسی اتصال قبل از استفاده
            pool_recycle=3600     # بازسازی اتصال بعد از ۱ ساعت
        )
        self.SessionLocal = sessionmaker(bind=self.engine)
    
    def create_tables(self):
        """ایجاد همه جدول‌ها"""
        Base.metadata.create_all(self.engine)
        logger.info("✅ Tables created successfully!")
        print("✅ Tables created successfully!")
    
    def get_session(self):
        """دریافت یک سشن دیتابیس"""
        return self.SessionLocal()
    
    def test_connection(self):
        """تست اتصال به دیتابیس"""
        try:
            session = self.get_session()
            session.execute(text("SELECT 1"))
            session.close()
            logger.info("✅ Database connection successful!")
            return True
        except Exception as e:
            logger.error(f"❌ Database connection error: {e}")
            return False
    def close(self):
        """بستن اتصال به دیتابیس"""
        try:
            self.engine.dispose()
            logger.info("✅ Database connection closed")
        except Exception as e:
            logger.error(f"❌ Error closing database connection: {e}")

db_manager = DatabaseManager()