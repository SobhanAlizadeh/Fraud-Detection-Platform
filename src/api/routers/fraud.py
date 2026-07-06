# src/api/routers/fraud.py
from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Optional
from datetime import datetime
import logging

from ..schemas.transaction import (
    TransactionRequest,
    TransactionResponse,
    BatchTransactionRequest,
    FraudStatsResponse,
    TransactionHistoryResponse,
    RiskProfileResponse,
    ModelPerformanceResponse,
    ErrorResponse
)
from ..services.fraud_service import FraudService

logger = logging.getLogger(__name__)

# ایجاد روتر
router = APIRouter(prefix="/api/v1/fraud", tags=["Fraud Detection"])

# ============================================
# وابستگی‌ها (Dependencies)
# ============================================

async def get_fraud_service():
    """تزریق سرویس تشخیص تقلب"""
    return FraudService()

# ============================================
# اندپوینت‌ها (Endpoints)
# ============================================

@router.post("/detect", response_model=TransactionResponse)
async def detect_fraud(
    transaction: TransactionRequest,
    service: FraudService = Depends(get_fraud_service)
):
    """
    🛡️ تشخیص تقلب برای یک تراکنش
    
    این اندپوینت یک تراکنش را دریافت کرده و با استفاده از مدل ML،
    احتمال تقلب بودن آن را محاسبه می‌کند.
    
    - **transaction_id**: شناسه یکتای تراکنش
    - **user_id**: شناسه کاربر
    - **amount**: مبلغ تراکنش
    - **timestamp**: زمان تراکنش
    - **merchant**: نام فروشنده (اختیاری)
    - **location**: موقعیت مکانی (اختیاری)
    """
    try:
        result = await service.detect_fraud(transaction.dict())
        return TransactionResponse(
            status="success",
            data=result,
            message="Fraud detection completed successfully",
            timestamp=datetime.utcnow()
        )
    except Exception as e:
        logger.error(f"Error in fraud detection: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch-detect", response_model=TransactionResponse)
async def batch_detect_fraud(
    request: BatchTransactionRequest,
    service: FraudService = Depends(get_fraud_service)
):
    """
    📦 تشخیص تقلب برای چند تراکنش به صورت دسته‌ای
    
    حداکثر ۱۰۰ تراکنش در هر درخواست مجاز است.
    """
    try:
        results = await service.batch_detect_fraud(
            [t.dict() for t in request.transactions]
        )
        return TransactionResponse(
            status="success",
            data=results,
            message=f"Batch detection completed for {len(request.transactions)} transactions",
            timestamp=datetime.utcnow()
        )
    except Exception as e:
        logger.error(f"Error in batch fraud detection: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history/{user_id}", response_model=TransactionHistoryResponse)
async def get_user_history(
    user_id: str,
    days: int = Query(30, ge=1, le=365, description="تعداد روزهای گذشته"),
    limit: int = Query(100, ge=1, le=1000, description="حداکثر تعداد تراکنش‌ها"),
    service: FraudService = Depends(get_fraud_service)
):
    """
    📜 دریافت تاریخچه تراکنش‌های یک کاربر
    
    - **user_id**: شناسه کاربر
    - **days**: تعداد روزهای گذشته (۱ تا ۳۶۵)
    - **limit**: حداکثر تعداد تراکنش‌ها (۱ تا ۱۰۰۰)
    """
    try:
        history = await service.get_transaction_history(user_id, days)
        
        # اعمال limit
        if limit and len(history) > limit:
            history = history[:limit]
        
        return TransactionHistoryResponse(
            status="success",
            data=history,
            user_id=user_id,
            days=days,
            total_count=len(history),
            timestamp=datetime.utcnow()
        )
    except Exception as e:
        logger.error(f"Error fetching history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history/all", response_model=TransactionHistoryResponse)
async def get_all_history(
    days: int = Query(7, ge=1, le=30),
    limit: int = Query(50, ge=1, le=500),
    service: FraudService = Depends(get_fraud_service)
):
    """دریافت تاریخچه همه تراکنش‌ها"""
    try:
        # ✅ فقط این متد صدا زده میشه
        history = await service.get_all_transactions(days, limit)
        
        return TransactionHistoryResponse(
            status="success",
            data=history,
            user_id="all",  # فقط برای نمایش - ربطی به کوئری نداره
            days=days,
            total_count=len(history),
            timestamp=datetime.utcnow()
        )
    except Exception as e:
        logger.error(f"Error fetching all history: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
@router.get("/stats", response_model=FraudStatsResponse)
async def get_fraud_stats(
    days: int = Query(7, ge=1, le=30, description="تعداد روزهای گذشته"),
    service: FraudService = Depends(get_fraud_service)
):
    """
    📊 دریافت آمار و ارقام مربوط به تقلب
    
    شامل:
    - تعداد کل تراکنش‌ها
    - تعداد تراکنش‌های تقلبی
    - نرخ تقلب
    - میانگین مبلغ
    - بیشترین مبلغ
    - آمار روزانه
    - فروشندگان پرتقلب
    """
    try:
        stats = await service.get_fraud_stats(days)
        return FraudStatsResponse(
            status="success",
            data=stats,
            timestamp=datetime.utcnow()
        )
    except Exception as e:
        logger.error(f"Error fetching stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/transaction/{transaction_id}")
async def get_transaction_details(
    transaction_id: str,
    service: FraudService = Depends(get_fraud_service)
):
    """
    🔍 دریافت جزئیات کامل یک تراکنش
    """
    try:
        result = await service.get_transaction_details(transaction_id)
        if result is None:
            raise HTTPException(status_code=404, detail="Transaction not found")
        
        return {
            "status": "success",
            "data": result,
            "timestamp": datetime.utcnow().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching transaction details: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/performance", response_model=ModelPerformanceResponse)
async def get_model_performance(
    service: FraudService = Depends(get_fraud_service)
):
    """
    📈 دریافت عملکرد مدل
    
    شامل:
    - تعداد کل پیش‌بینی‌ها
    - تعداد تقلب‌های شناسایی شده
    - نرخ تقلب
    - میانگین احتمال تقلب
    - توزیع احتمالات
    """
    try:
        performance = await service.get_model_performance()
        return ModelPerformanceResponse(
            status="success",
            data=performance,
            timestamp=datetime.utcnow()
        )
    except Exception as e:
        logger.error(f"Error fetching model performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/risk-profile/{user_id}", response_model=RiskProfileResponse)
async def get_user_risk_profile(
    user_id: str,
    service: FraudService = Depends(get_fraud_service)
):
    """
    👤 دریافت پروفایل ریسک کاربر
    
    شامل:
    - امتیاز ریسک (۰ تا ۱۰۰)
    - سطح ریسک (LOW, MEDIUM, HIGH)
    - متریک‌های تراکنش
    - فاکتورهای ریسک
    """
    try:
        profile = await service.get_user_risk_profile(user_id)
        return RiskProfileResponse(
            status="success",
            data=profile,
            timestamp=datetime.utcnow()
        )
    except Exception as e:
        logger.error(f"Error fetching risk profile: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/update-threshold")
async def update_threshold(
    threshold: float = Query(..., ge=0.1, le=0.9, description="آستانه جدید (بین ۰.۱ و ۰.۹)"),
    service: FraudService = Depends(get_fraud_service)
):
    """
    🎯 به‌روزرسانی آستانه تشخیص تقلب
    
    آستانه بالاتر = تشخیص دقیق‌تر اما ممکن است برخی تقلب‌ها را از دست بدهد
    آستانه پایین‌تر = تشخیص بیشتر اما با احتمال خطای بالاتر
    """
    try:
        result = await service.update_threshold(threshold)
        return {
            "status": "success",
            "data": result,
            "message": f"Threshold updated to {threshold}",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error updating threshold: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def fraud_health():
    """
    🔍 بررسی سلامت ماژول تشخیص تقلب
    """
    return {
        "status": "healthy",
        "module": "fraud_detection",
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/threshold")
async def get_current_threshold(
    service: FraudService = Depends(get_fraud_service)
):
    """
    📊 دریافت آستانه فعلی تشخیص
    """
    return {
        "status": "success",
        "threshold": service.threshold,
        "timestamp": datetime.utcnow().isoformat()
    }

@router.get("/result/{transaction_id}")
async def get_fraud_result(
    transaction_id: str,
    service: FraudService = Depends(get_fraud_service)
):
    """دریافت نتیجه نهایی تشخیص تقلب برای یک تراکنش"""
    try:
        result = await service.get_transaction_result(transaction_id)
        if result is None:
            raise HTTPException(status_code=404, detail="Transaction not found")
        return {
            "status": "success",
            "data": result,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error fetching result: {e}")
        raise HTTPException(status_code=500, detail=str(e))