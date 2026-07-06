# src/api/services/fraud_service.py
from typing import Dict, Any, List, Optional
import pandas as pd
from datetime import datetime, timedelta
import logging
from sqlalchemy import func, and_

from ...core.database import db_manager, Transaction, Prediction
from ...core.config import config
from ...ml_engineering.inference import FraudInferenceService  
from src.core.kafka_client import kafka_client

logger = logging.getLogger(__name__)

class FraudService:
    """سرویس اصلی تشخیص تقلب"""
    
    def __init__(self):
        self.inference_service = FraudInferenceService()  # <--- درستش کن
        self.threshold = 0.5
        self.threshold = 0.5
    
    async def detect_fraud(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """
        تشخیص تقلب برای یک تراکنش با استفاده از مدل ML
        
        Args:
            transaction: دیکشنری حاوی اطلاعات تراکنش
            
        Returns:
            نتیجه تشخیص شامل احتمال تقلب و وضعیت
        """
        try:
            # 1. ذخیره تراکنش در دیتابیس
            saved_transaction = self._save_transaction(transaction)
            logger.info(f"✅ Transaction saved: {transaction.get('transaction_id')}")
            
            # 2. انتشار در کافکا
            kafka_client.publish(
                topic="transactions",
                message=transaction,
                key=transaction.get('transaction_id')
            )
            logger.info(f"📤 Transaction published to Kafka: {transaction.get('transaction_id')}")
            
            # 3. برگرداندن پاسخ اولیه
            return {
                'transaction_id': transaction.get('transaction_id'),
                'user_id': transaction.get('user_id'),
                'amount': transaction.get('amount'),
                'timestamp': transaction.get('timestamp'),
                'status': 'pending',  # در حال پردازش
                'message': 'Transaction received and queued for fraud detection'
            }
        
        except Exception as e:
            logger.error(f"❌ Error in fraud detection: {e}")
            return {
                'error': str(e),
                'transaction_id': transaction.get('transaction_id'),
                'status': 'failed'
            }
    
    def _save_transaction(self, transaction: Dict[str, Any]) -> Transaction:
        """ذخیره تراکنش در دیتابیس"""
        session = db_manager.get_session()
        try:
            user_id = transaction.get('user_id', '0')
            user_id_int = self._extract_user_id(user_id)
            
            # ایجاد تراکنش جدید
            new_transaction = Transaction(
                transaction_id=transaction['transaction_id'],
                user_id=user_id_int,
                customer_id=user_id_int,
                merchant_id=1,
                merchant=transaction.get('merchant'),
                amount=transaction['amount'],
                transaction_type=transaction.get('transaction_type', 'online'),
                timestamp=datetime.fromisoformat(transaction['timestamp']) if isinstance(transaction['timestamp'], str) else transaction['timestamp'],
                is_fraud=0,
                location=transaction.get('location'),
                device_id=transaction.get('device_id'),
                ip_address=transaction.get('ip_address'),
                fraud_score=0.0
            )
            
            session.add(new_transaction)
            session.commit()
            session.refresh(new_transaction)
            return new_transaction
            
        except Exception as e:
            session.rollback()
            logger.error(f"❌ Error saving transaction: {e}")
            raise
        finally:
            session.close()
    
    def _update_transaction_fraud(self, transaction_id: str, is_fraud: bool, fraud_score: float):
        """به‌روزرسانی وضعیت تقلب تراکنش"""
        session = db_manager.get_session()
        try:
            transaction = session.query(Transaction).filter(
                Transaction.transaction_id == transaction_id
            ).first()
            
            if transaction:
                transaction.is_fraud = 1 if is_fraud else 0
                transaction.fraud_score = fraud_score
                session.commit()
                logger.info(f"✅ Updated fraud status for {transaction_id}")
                
        except Exception as e:
            session.rollback()
            logger.error(f"❌ Error updating fraud status: {e}")
        finally:
            session.close()
            
    async def batch_detect_fraud(self, transactions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """تشخیص دسته‌ای"""
        results = []
        for t in transactions:
            result = await self.detect_fraud(t)
            results.append(result)
        return results
    
    def _extract_user_id(self, user_id: str) -> int:
        """استخراج عدد از user_id مثل 'user_3851' -> 3851"""
        if isinstance(user_id, int):
            return user_id
        try:
            # اگر فرمت user_3851 بود
            if user_id.startswith('user_'):
                return int(user_id.split('_')[1])
            # اگر فقط عدد بود
            return int(user_id)
        except:
            # مقدار پیش‌فرض
            return 0

    async def get_all_transactions(self, days: int = 7, limit: int = 50) -> List[Dict[str, Any]]:
        """دریافت همه تراکنش‌های اخیر (بدون فیلتر کاربر)"""
        session = db_manager.get_session()
        try:
            start_date = datetime.utcnow() - timedelta(days=days)
            
            # ✅ کوئری بدون user_id
            transactions = session.query(Transaction).filter(
                Transaction.timestamp >= start_date
            ).order_by(
                Transaction.timestamp.desc()
            ).limit(limit).all()
            
            result = []
            for t in transactions:
                result.append({
                    'transaction_id': t.transaction_id,
                    'user_id': t.user_id,
                    'customer_id': t.customer_id,
                    'amount': t.amount,
                    'timestamp': t.timestamp.isoformat(),
                    'merchant': t.merchant,
                    'location': t.location,
                    'transaction_type': t.transaction_type,
                    'is_fraud': t.is_fraud,
                    'fraud_score': t.fraud_score
                })
            
            logger.info(f"✅ Retrieved {len(result)} transactions (all users)")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error in get_all_transactions: {e}")
            raise
        finally:
            session.close()
    
    async def get_transaction_history(self, user_id: str, days: int = 30) -> List[Dict[str, Any]]:
        """دریافت تاریخچه تراکنش‌های یک کاربر خاص"""
        session = db_manager.get_session()
        try:
            start_date = datetime.utcnow() - timedelta(days=days)
            
            # تبدیل user_id به عدد (اگر عددی باشه)
            try:
                user_id_int = int(user_id)
            except ValueError:
                logger.error(f"Invalid user_id: {user_id}")
                return []
            
            transactions = session.query(Transaction).filter(
                Transaction.user_id == user_id_int,
                Transaction.timestamp >= start_date
            ).order_by(
                Transaction.timestamp.desc()
            ).limit(100).all()
            
            return [
                {
                    'transaction_id': t.transaction_id,
                    'user_id': t.user_id,
                    'customer_id': t.customer_id,
                    'amount': t.amount,
                    'timestamp': t.timestamp.isoformat(),
                    'merchant': t.merchant,
                    'location': t.location,
                    'transaction_type': t.transaction_type,
                    'is_fraud': t.is_fraud,
                    'fraud_score': t.fraud_score
                }
                for t in transactions
            ]
        finally:
            session.close()
    
    async def get_fraud_stats(self, days: int = 7) -> Dict[str, Any]:
        """دریافت آمار و ارقام مربوط به تقلب"""
        session = db_manager.get_session()
        
        try:
            start_date = datetime.utcnow() - timedelta(days=days)
            
            # 1. تعداد کل تراکنش‌ها
            total = session.query(Transaction).filter(
                Transaction.timestamp >= start_date
            ).count()
            
            # 2. تعداد تراکنش‌های تقلبی
            fraud_count = session.query(Transaction).filter(
                Transaction.timestamp >= start_date,
                Transaction.is_fraud == True
            ).count()
            
            # 3. میانگین مبلغ
            avg_result = session.query(func.avg(Transaction.amount)).filter(
                Transaction.timestamp >= start_date
            ).scalar()
            avg_amount = float(avg_result) if avg_result is not None else 0.0
            
            # 4. بیشترین مبلغ
            max_result = session.query(func.max(Transaction.amount)).filter(
                Transaction.timestamp >= start_date
            ).scalar()
            max_amount = float(max_result) if max_result is not None else 0.0
            
            # 5. آمار روزانه (با استفاده از func.date)
            try:
                daily_stats_raw = session.query(
                    func.date(Transaction.timestamp).label('date'),
                    func.count().label('count'),
                    func.sum(Transaction.amount).label('total_amount'),
                    func.sum(func.cast(Transaction.is_fraud, func.Integer())).label('fraud_count')
                ).filter(
                    Transaction.timestamp >= start_date
                ).group_by(
                    func.date(Transaction.timestamp)
                ).order_by(
                    func.date(Transaction.timestamp)
                ).all()
                
                daily_stats = []
                for stat in daily_stats_raw:
                    # تبدیل ایمن تاریخ
                    date_str = str(stat.date) if stat.date else None
                    daily_stats.append({
                        'date': date_str,
                        'count': int(stat.count) if stat.count else 0,
                        'total_amount': float(stat.total_amount) if stat.total_amount else 0.0,
                        'fraud_count': int(stat.fraud_count) if stat.fraud_count else 0
                    })
            except Exception as e:
                logger.warning(f"Error in daily stats: {e}")
                daily_stats = []
            
            # 6. پرتکرارترین فروشنده‌ها در تراکنش‌های تقلبی
            try:
                top_merchants_raw = session.query(
                    Transaction.merchant,
                    func.count().label('count')
                ).filter(
                    Transaction.timestamp >= start_date,
                    Transaction.is_fraud == True,
                    Transaction.merchant.isnot(None)
                ).group_by(
                    Transaction.merchant
                ).order_by(
                    func.count().desc()
                ).limit(5).all()
                
                top_fraud_merchants = [
                    {'merchant': m.merchant, 'count': int(m.count)}
                    for m in top_merchants_raw
                ]
            except Exception as e:
                logger.warning(f"Error in merchant stats: {e}")
                top_fraud_merchants = []
            
            return {
                'total_transactions': total,
                'fraud_transactions': fraud_count,
                'fraud_rate': round(fraud_count / total * 100, 2) if total > 0 else 0,
                'avg_amount': round(avg_amount, 2),
                'max_amount': round(max_amount, 2),
                'period_days': days,
                'daily_stats': daily_stats,
                'top_fraud_merchants': top_fraud_merchants
            }
            
        except Exception as e:
            logger.error(f"Error in get_fraud_stats: {e}")
            # برگرداندن داده‌های پیش‌فرض در صورت خطا
            return {
                'total_transactions': 0,
                'fraud_transactions': 0,
                'fraud_rate': 0,
                'avg_amount': 0,
                'max_amount': 0,
                'period_days': days,
                'daily_stats': [],
                'top_fraud_merchants': [],
                'error': str(e)
            }
        finally:
            session.close()
    
    async def get_transaction_details(self, transaction_id: str) -> Optional[Dict[str, Any]]:
        """دریافت جزئیات یک تراکنش"""
        session = db_manager.get_session()
        try:
            transaction = session.query(Transaction).filter(
                Transaction.transaction_id == transaction_id
            ).first()
            
            if not transaction:
                return None
            
            return {
                'transaction_id': transaction.transaction_id,
                'user_id': transaction.user_id,
                'amount': float(transaction.amount) if transaction.amount else 0,
                'timestamp': transaction.timestamp.isoformat() if transaction.timestamp else None,
                'merchant': transaction.merchant,
                'location': transaction.location,
                'is_fraud': bool(transaction.is_fraud) if transaction.is_fraud is not None else False,
                'fraud_score': float(transaction.fraud_score) if transaction.fraud_score is not None else 0.0
            }
        finally:
            session.close()
    
    async def get_model_performance(self) -> Dict[str, Any]:
        """دریافت عملکرد مدل"""
        session = db_manager.get_session()
        try:
            predictions = session.query(Prediction).order_by(
                Prediction.id.desc()
            ).limit(1000).all()
            
            if not predictions:
                return {'message': 'No predictions available'}
            
            total = len(predictions)
            fraud_predicted = sum(1 for p in predictions if p.is_fraud_predicted)
            avg_probability = sum(p.fraud_probability for p in predictions) / total if total > 0 else 0
            
            probability_distribution = {
                'low': sum(1 for p in predictions if p.fraud_probability < 0.3),
                'medium': sum(1 for p in predictions if 0.3 <= p.fraud_probability < 0.7),
                'high': sum(1 for p in predictions if p.fraud_probability >= 0.7)
            }
            
            return {
                'total_predictions': total,
                'fraud_predicted': fraud_predicted,
                'fraud_rate_predicted': round(fraud_predicted / total * 100, 2) if total > 0 else 0,
                'avg_fraud_probability': round(avg_probability, 3),
                'probability_distribution': probability_distribution,
                'latest_prediction_time': predictions[0].prediction_time.isoformat() if predictions else None
            }
        finally:
            session.close()
    
    async def get_user_risk_profile(self, user_id: str) -> Dict[str, Any]:
        """دریافت پروفایل ریسک کاربر"""
        session = db_manager.get_session()
        try:
            start_date = datetime.utcnow() - timedelta(days=30)
            
            transactions = session.query(Transaction).filter(
                Transaction.user_id == user_id,
                Transaction.timestamp >= start_date
            ).all()
            
            if not transactions:
                return {'message': 'No transactions found for this user'}
            
            total_amount = sum(float(t.amount) if t.amount else 0 for t in transactions)
            avg_amount = total_amount / len(transactions) if transactions else 0
            max_amount = max(float(t.amount) if t.amount else 0 for t in transactions)
            fraud_count = sum(1 for t in transactions if t.is_fraud)
            
            risk_score = 0
            risk_factors = []
            
            if len(transactions) > 20:
                risk_score += 10
                risk_factors.append({'factor': 'high_transaction_count', 'weight': 10})
            
            if max_amount > avg_amount * 3:
                risk_score += 20
                risk_factors.append({'factor': 'abnormal_amount', 'weight': 20})
            
            if fraud_count > 0:
                risk_score += 30
                risk_factors.append({'factor': 'fraud_history', 'weight': 30})
            
            risk_score = min(risk_score, 100)
            
            if risk_score >= 70:
                risk_level = 'HIGH'
            elif risk_score >= 40:
                risk_level = 'MEDIUM'
            else:
                risk_level = 'LOW'
            
            return {
                'user_id': user_id,
                'risk_score': risk_score,
                'risk_level': risk_level,
                'metrics': {
                    'total_transactions': len(transactions),
                    'total_amount': round(total_amount, 2),
                    'avg_amount': round(avg_amount, 2),
                    'max_amount': round(max_amount, 2),
                    'fraud_transactions': fraud_count,
                    'fraud_rate': round(fraud_count / len(transactions) * 100, 2) if transactions else 0
                },
                'risk_factors': risk_factors
            }
        finally:
            session.close()
    
    async def update_threshold(self, threshold: float) -> Dict[str, Any]:
        """به‌روزرسانی آستانه"""
        old = self.threshold
        self.threshold = threshold
        return {
            'old_threshold': old,
            'new_threshold': threshold,
            'status': 'updated'
        }
async def get_threshold(self) -> Dict[str, Any]:
    """دریافت آستانه فعلی"""
    return {
        "threshold": self.threshold,
        "timestamp": datetime.utcnow().isoformat()
    }