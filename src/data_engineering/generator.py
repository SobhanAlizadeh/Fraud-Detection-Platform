# src/data_engineering/generator.py
import pandas as pd
import numpy as np
from faker import Faker
from pathlib import Path
import os

fake = Faker()
Faker.seed(42)
np.random.seed(42)

# پیدا کردن مسیر ریشه پروژه
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_RAW_DIR = BASE_DIR / "data" / "raw"

# --- ۱. تولید دیتای مشتریان ---
def generate_customers(n=1000):
    customers = []
    for cust_id in range(1, n + 1):
        customers.append({
            'customer_id': cust_id,
            'name': fake.name(),
            'age': np.random.randint(18, 75),
            'job': fake.job(),
            'city': fake.city(),
            'account_age_days': np.random.randint(10, 3650)
        })
    return pd.DataFrame(customers)

# --- ۲. تولید دیتای پذیرنده‌ها ---
def generate_merchants(n=500):
    merchants = []
    for merch_id in range(1, n + 1):
        merchants.append({
            'merchant_id': merch_id,
            'merchant_name': fake.company(),
            'category': np.random.choice(['Grocery', 'Electronics', 'Travel', 'Restaurant', 'Crypto', 'Jewelry']),
            'risk_level': np.random.choice([1, 2, 3], p=[0.7, 0.2, 0.1])
        })
    return pd.DataFrame(merchants)

# --- ۳. تولید تراکنش‌ها ---
def generate_transactions(customers, merchants, n=50000):
    transactions = []
    for _ in range(n):
        cust = customers.sample(1).iloc[0]
        merch = merchants.sample(1).iloc[0]
        amount = round(np.random.exponential(scale=100), 2)
        is_fraud = 0
        
        # قوانین تشخیص تقلب
        if amount > 800 and merch['risk_level'] == 3 and cust['account_age_days'] < 180:
            is_fraud = 1
        elif amount > 500 and merch['category'] == 'Electronics' and amount.is_integer():
            is_fraud = 1 if np.random.rand() > 0.7 else 0
        elif np.random.rand() < 0.001:
            is_fraud = 1

        transactions.append({
            'transaction_id': fake.uuid4(),
            'customer_id': cust['customer_id'],  # <--- اضافه شد
            'user_id': cust['customer_id'],  # <--- اضافه شد
            'merchant_id': merch['merchant_id'],
            'merchant': merch['merchant_name'],  # <--- اضافه شد
            'amount': amount,
            'transaction_type': np.random.choice(['POS', 'Online', 'ATM', 'Transfer']),
            'timestamp': fake.date_time_between(start_date='-1y', end_date='now'),
            'is_fraud': is_fraud,
            'location': fake.city(),  # <--- اضافه شد
            'device_id': f"DEV{np.random.randint(1000, 9999)}",  # <--- اضافه شد
            'ip_address': fake.ipv4(),  # <--- اضافه شد
            'fraud_score': round(np.random.uniform(0.1, 0.9) if is_fraud else np.random.uniform(0.01, 0.3), 3)
        })
        
    return pd.DataFrame(transactions)

if __name__ == "__main__":
    print("=" * 50)
    print("📊 Generating custom datasets...")
    print("=" * 50)
    
    # ساخت پوشه
    os.makedirs(DATA_RAW_DIR, exist_ok=True)
    
    # تولید
    customers_df = generate_customers()
    merchants_df = generate_merchants()
    transactions_df = generate_transactions(customers_df, merchants_df, n=50000)
    
    # ذخیره
    customers_df.to_csv(DATA_RAW_DIR / 'customers.csv', index=False)
    merchants_df.to_csv(DATA_RAW_DIR / 'merchants.csv', index=False)
    transactions_df.to_csv(DATA_RAW_DIR / 'transactions.csv', index=False)
    
    print(f"✅ Generated {len(customers_df)} customers.")
    print(f"✅ Generated {len(merchants_df)} merchants.")
    print(f"✅ Generated {len(transactions_df)} transactions.")
    print(f"   Fraud count: {transactions_df['is_fraud'].sum()}")
    print(f"   Fraud rate: {transactions_df['is_fraud'].mean()*100:.2f}%")
    print(f"\n📁 Files saved in: {DATA_RAW_DIR}")
    print("=" * 50)