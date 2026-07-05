#!/bin/bash
# ============================================
# اسکریپت ورودی برای کانتینر FastAPI
# ============================================

set -e

echo "🚀 Starting Fraud Detection API Container..."

# تنظیمات لاگ
LOG_DIR="/app/logs"
mkdir -p $LOG_DIR

# تابع برای لاگ کردن با timestamp
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# 1. انتظار برای PostgreSQL
log_message "⏳ Waiting for PostgreSQL to be ready..."
until PGPASSWORD=$DB_PASSWORD psql -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" -c '\q' 2>/dev/null; do
    log_message "⏳ PostgreSQL is unavailable - sleeping..."
    sleep 2
done
log_message "✅ PostgreSQL is ready!"

# 2. ایجاد دایرکتوری‌های مورد نیاز
log_message "📁 Creating required directories..."
mkdir -p /app/data/{raw,processed,features}
mkdir -p /app/models
mkdir -p /app/monitoring_reports

# 3. بررسی وجود مدل
if [ ! -f "/app/models/best_model.pkl" ]; then
    log_message "⚠️ No model found in /app/models/best_model.pkl"
    log_message "⚠️ The API will run in development mode without a pre-trained model"
fi

# 4. اجرای پیاده‌سازی (Entrypoint)
log_message "✅ Container initialization complete!"

echo "📦 Installing joblib..."
pip install joblib==1.3.2

# اجرای دستور ورودی (CMD)
exec "$@"