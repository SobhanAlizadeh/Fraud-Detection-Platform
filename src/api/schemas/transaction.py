from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

class TransactionType(str, Enum):
    """انواع تراکنش"""
    ONLINE = "online"
    IN_STORE = "in_store"
    ATM = "atm"
    TRANSFER = "transfer"
    PAYMENT = "payment"

class TransactionStatus(str, Enum):
    """وضعیت تراکنش"""
    PENDING = "pending"
    PROCESSED = "processed"
    FLAGGED = "flagged"
    APPROVED = "approved"
    REJECTED = "rejected"

# ============== مدل‌های درخواست ==============

class TransactionRequest(BaseModel):
    """مدل درخواست تشخیص تقلب برای یک تراکنش"""
    transaction_id: str = Field(..., description="شناسه یکتای تراکنش", example="TX20240115001")
    user_id: str = Field(..., description="شناسه کاربر", example="user_12345")
    amount: float = Field(..., gt=0, description="مبلغ تراکنش", example=1250.50)
    timestamp: datetime = Field(..., description="زمان تراکنش", example="2024-01-15T10:30:00")
    merchant: Optional[str] = Field(None, description="نام فروشنده", example="Amazon")
    location: Optional[str] = Field(None, description="موقعیت مکانی", example="Tehran")
    transaction_type: Optional[TransactionType] = Field(None, description="نوع تراکنش")
    device_id: Optional[str] = Field(None, description="شناسه دستگاه", example="DEV-7890")
    ip_address: Optional[str] = Field(None, description="آدرس IP", example="192.168.1.100")
    features: Optional[Dict[str, Any]] = Field(None, description="فیچرهای اضافی")
    
    @validator('amount')
    def validate_amount(cls, v):
        if v <= 0:
            raise ValueError("مبلغ باید بزرگتر از صفر باشد")
        return v
    
    @validator('transaction_id')
    def validate_transaction_id(cls, v):
        if not v or len(v) < 3:
            raise ValueError("شناسه تراکنش معتبر نیست")
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "transaction_id": "TX20240115001",
                "user_id": "user_12345",
                "amount": 1250.50,
                "timestamp": "2024-01-15T10:30:00",
                "merchant": "Amazon",
                "location": "Tehran",
                "transaction_type": "online",
                "device_id": "DEV-7890",
                "ip_address": "192.168.1.100"
            }
        }

class BatchTransactionRequest(BaseModel):
    """مدل درخواست تشخیص تقلب برای چند تراکنش"""
    transactions: List[TransactionRequest] = Field(..., description="لیست تراکنش‌ها", min_items=1, max_items=100)
    
    @validator('transactions')
    def validate_batch(cls, v):
        if len(v) > 100:
            raise ValueError("حداکثر 100 تراکنش در هر درخواست مجاز است")
        # بررسی تکراری نبودن transaction_id
        ids = [t.transaction_id for t in v]
        if len(ids) != len(set(ids)):
            raise ValueError("شناسه تراکنش‌ها نباید تکراری باشد")
        return v

class ThresholdUpdateRequest(BaseModel):
    """مدل درخواست به‌روزرسانی آستانه"""
    threshold: float = Field(..., ge=0.0, le=1.0, description="آستانه جدید (بین 0 و 1)", example=0.7)
    
    @validator('threshold')
    def validate_threshold(cls, v):
        if v < 0 or v > 1:
            raise ValueError("آستانه باید بین 0 و 1 باشد")
        return v

class DateRangeRequest(BaseModel):
    """مدل درخواست بازه زمانی"""
    start_date: Optional[datetime] = Field(None, description="تاریخ شروع")
    end_date: Optional[datetime] = Field(None, description="تاریخ پایان")
    days: Optional[int] = Field(7, ge=1, le=365, description="تعداد روزهای گذشته")
    
    @validator('end_date')
    def validate_dates(cls, v, values):
        if v and values.get('start_date') and v < values['start_date']:
            raise ValueError("تاریخ پایان باید بعد از تاریخ شروع باشد")
        return v

# ============== مدل‌های پاسخ ==============

class PredictionResult(BaseModel):
    """مدل نتیجه پیش‌بینی"""
    fraud_probability: float = Field(..., description="احتمال تقلب (0 تا 1)")
    is_fraud: bool = Field(..., description="آیا تراکنش تقلبی است؟")
    threshold: float = Field(..., description="آستانه استفاده شده")
    model_version: str = Field(..., description="نسخه مدل")
    prediction_time: str = Field(..., description="زمان پیش‌بینی")
    features_used: Optional[List[str]] = Field(None, description="فیچرهای استفاده شده")
    feature_importance: Optional[Dict[str, float]] = Field(None, description="اهمیت فیچرها")

class TransactionResponse(BaseModel):
    """مدل پاسخ تشخیص تقلب"""
    status: str = Field(..., description="وضعیت درخواست", example="success")
    data: Dict[str, Any] = Field(..., description="داده‌های پاسخ")
    message: Optional[str] = Field(None, description="پیام تکمیلی")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="زمان پاسخ")

class TransactionHistoryResponse(BaseModel):
    """مدل پاسخ تاریخچه تراکنش‌ها"""
    status: str = Field(..., description="وضعیت درخواست")
    data: List[Dict[str, Any]] = Field(..., description="لیست تراکنش‌ها")
    user_id: str = Field(..., description="شناسه کاربر")
    days: int = Field(..., description="تعداد روزهای بررسی شده")
    total_count: int = Field(..., description="تعداد کل تراکنش‌ها")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class FraudStatsResponse(BaseModel):
    """مدل پاسخ آمار تقلب"""
    status: str = Field(..., description="وضعیت درخواست")
    data: Dict[str, Any] = Field(..., description="داده‌های آماری")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class RiskProfileResponse(BaseModel):
    """مدل پاسخ پروفایل ریسک کاربر"""
    status: str = Field(..., description="وضعیت درخواست")
    data: Dict[str, Any] = Field(..., description="داده‌های پروفایل ریسک")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class ModelPerformanceResponse(BaseModel):
    """مدل پاسخ عملکرد مدل"""
    status: str = Field(..., description="وضعیت درخواست")
    data: Dict[str, Any] = Field(..., description="داده‌های عملکرد مدل")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class ErrorResponse(BaseModel):
    """مدل پاسخ خطا"""
    status: str = Field("error", description="وضعیت")
    error: str = Field(..., description="پیام خطا")
    detail: Optional[str] = Field(None, description="جزئیات خطا")
    transaction_id: Optional[str] = Field(None, description="شناسه تراکنش")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

# ============== مدل‌های داخلی ==============

class TransactionInternal(BaseModel):
    """مدل داخلی تراکنش برای استفاده در سرویس‌ها"""
    transaction_id: str
    user_id: str
    amount: float
    timestamp: datetime
    merchant: Optional[str] = None
    location: Optional[str] = None
    transaction_type: Optional[str] = None
    device_id: Optional[str] = None
    ip_address: Optional[str] = None
    features: Dict[str, Any] = {}
    is_fraud: Optional[bool] = None
    fraud_score: Optional[float] = None
    created_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True

class PredictionInternal(BaseModel):
    """مدل داخلی پیش‌بینی برای استفاده در سرویس‌ها"""
    transaction_id: str
    fraud_probability: float
    is_fraud_predicted: bool
    model_version: str
    features_used: List[str]
    feature_importance: Dict[str, float]
    prediction_time: datetime
    
    class Config:
        from_attributes = True

# ============== مدل‌های Webhook ==============

class WebhookPayload(BaseModel):
    """مدل پیلود Webhook برای ارسال به سیستم‌های خارجی"""
    event_type: str = Field(..., description="نوع رویداد", example="fraud_detected")
    transaction_id: str = Field(..., description="شناسه تراکنش")
    user_id: str = Field(..., description="شناسه کاربر")
    fraud_probability: float = Field(..., description="احتمال تقلب")
    is_fraud: bool = Field(..., description="نتیجه تشخیص")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Optional[Dict[str, Any]] = Field(None, description="اطلاعات اضافی")

# ============== مدل‌های Dashboard ==============

class DashboardSummary(BaseModel):
    """مدل خلاصه داشبورد"""
    total_transactions_24h: int
    fraud_transactions_24h: int
    fraud_rate_24h: float
    avg_response_time_ms: float
    active_models: List[str]
    system_health: str
    last_prediction_time: Optional[datetime]
    
class AlertRule(BaseModel):
    """مدل قوانین هشدار"""
    rule_id: str
    name: str
    condition: str
    threshold: float
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    enabled: bool
    created_at: datetime
    updated_at: Optional[datetime] = None