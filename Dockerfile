# ============================================
# Dockerfile ساده‌تر با نصب مرحله‌ای
# ============================================

FROM python:3.11-slim

WORKDIR /app

# نصب پیش‌نیازها
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    python3-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# کپی و نصب وابستگی‌ها
COPY requirements.txt .

# نصب با --no-cache و --upgrade
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

RUN pip install --no-cache-dir python-dateutil pytz six 

# نصب پکیج‌های اصلی با --no-deps برای جلوگیری از conflict
RUN pip install --no-cache-dir numpy pandas scikit-learn xgboost lightgbm catboost --no-deps

RUN RUN pip install --no-cache-dir scipy==1.11.4

RUN pip install --no-cache-dir joblib==1.3.2 scikit-learn==1.3.2
# نصب بقیه وابستگی‌ها
RUN pip install --no-cache-dir -r requirements.txt --no-deps || true

# نصب وابستگی‌های باقی‌مانده
RUN pip install --no-cache-dir fastapi uvicorn sqlalchemy psycopg2-binary python-dotenv

# کپی سورس
COPY . .

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]