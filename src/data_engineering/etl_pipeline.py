# src/data_engineering/etl_pipeline.py
import pandas as pd
from sqlalchemy import create_engine
from pathlib import Path
import os

# تنظیمات اتصال به دیتابیس (با متغیرهای محیطی)
DB_USER = os.getenv("DB_USER", "fraud_user")
DB_PASS = os.getenv("DB_PASSWORD", "fraud_pass")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "fraud_db")

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

def get_data_path():
    """دریافت مسیر دایرکتوری داده"""
    base_dir = Path(__file__).resolve().parent.parent.parent
    return base_dir / "data" / "raw"

def extract():
    """خواندن داده‌ها از فایل‌های CSV"""
    print("🔄 Extracting data...")
    data_path = get_data_path()
    
    customers = pd.read_csv(data_path / 'customers.csv')
    merchants = pd.read_csv(data_path / 'merchants.csv')
    transactions = pd.read_csv(data_path / 'transactions.csv')
    
    print(f"   ✅ Customers: {len(customers)}")
    print(f"   ✅ Merchants: {len(merchants)}")
    print(f"   ✅ Transactions: {len(transactions)}")
    return customers, merchants, transactions

def transform(customers, merchants, transactions):
    """پاکسازی و تبدیل داده‌ها"""
    print("🔄 Transforming data...")
    
    # تبدیل timestamp
    transactions['timestamp'] = pd.to_datetime(transactions['timestamp'])
    
    # حذف مقادیر خالی
    transactions = transactions.dropna()
    customers = customers.dropna()
    merchants = merchants.dropna()
    
    # تبدیل is_fraud به int
    transactions['is_fraud'] = transactions['is_fraud'].astype(int)
    
    print(f"   ✅ Transactions after cleaning: {len(transactions)}")
    return customers, merchants, transactions

def load(customers, merchants, transactions):
    """نوشتن داده‌ها در PostgreSQL"""
    print("🔄 Loading data into PostgreSQL...")
    engine = create_engine(DATABASE_URL)
    
    try:
        # بارگذاری به ترتیب (برای رعایت foreign key)
        customers.to_sql('customers', engine, if_exists='replace', index=False)
        print("   ✅ Customers loaded")
        
        merchants.to_sql('merchants', engine, if_exists='replace', index=False)
        print("   ✅ Merchants loaded")
        
        transactions.to_sql('transactions', engine, if_exists='replace', index=False)
        print("   ✅ Transactions loaded")
        
        print("✅ Data loaded successfully!")
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        raise

def run_etl():
    """اجرای کل پایپلاین"""
    print("=" * 50)
    print("🚀 Starting ETL Pipeline")
    print("=" * 50)
    
    # 1. Extract
    customers, merchants, transactions = extract()
    
    # 2. Transform
    customers, merchants, transactions = transform(customers, merchants, transactions)
    
    # 3. Load
    load(customers, merchants, transactions)
    
    print("=" * 50)
    print("✅ ETL Pipeline Complete!")
    print("=" * 50)

if __name__ == "__main__":
    run_etl()