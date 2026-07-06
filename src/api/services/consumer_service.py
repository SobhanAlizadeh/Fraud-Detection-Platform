import logging
from typing import Any, Dict
from src.core.kafka_client import kafka_client
from src.ml_engineering.inference import FraudInferenceService
from src.core.database import db_manager, Transaction, Prediction
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class FraudConsumerService:
    def __init__(self):
        self.inference_service = FraudInferenceService()
        self.running = False
        self.consumer = None
        
    def process_transaction(self, transaction: Dict[str, Any]):
        """پردازش تراکنش دریافتی از کافکا"""
        try:
            logger.info(f"🔄 Processing transaction: {transaction.get('transaction_id')}")
            
            # 1. پیش‌بینی با مدل
            prediction = self.inference_service.predict(transaction)
            
            # 2. ذخیره نتیجه در دیتابیس
            self._save_prediction(transaction, prediction)
            
            # 3. به‌روزرسانی وضعیت تراکنش
            self._update_transaction(transaction.get('transaction_id'), prediction)
            
            logger.info(f"✅ Transaction {transaction.get('transaction_id')} processed: fraud={prediction.get('is_fraud')}")
            
        except Exception as e:
            logger.error(f"❌ Error processing transaction: {e}")
    
    def _save_prediction(self, transaction: Dict[str, Any], prediction: Dict[str, Any]):
        session = db_manager.get_session()
        try:
            new_prediction = Prediction(
                transaction_id=transaction.get('transaction_id'),
                fraud_probability=prediction.get('fraud_probability', 0.0),
                is_fraud_predicted=prediction.get('is_fraud', False),
                model_version=prediction.get('model_version', '1.0.0'),
                features_used=prediction.get('features_used', [])
            )
            session.add(new_prediction)
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"❌ Error saving prediction: {e}")
        finally:
            session.close()
    
    def _update_transaction(self, transaction_id: str, prediction: Dict[str, Any]):
        session = db_manager.get_session()
        try:
            transaction = session.query(Transaction).filter(
                Transaction.transaction_id == transaction_id
            ).first()
            if transaction:
                transaction.is_fraud = 1 if prediction.get('is_fraud', False) else 0
                transaction.fraud_score = prediction.get('fraud_probability', 0.0)
                session.commit()
                logger.info(f"✅ Updated transaction {transaction_id} with fraud status")
        except Exception as e:
            session.rollback()
            logger.error(f"❌ Error updating transaction: {e}")
        finally:
            session.close()
    
    def start(self):
        """شروع مصرف‌کننده"""
        if self.running:
            logger.warning("⚠️ Consumer already running")
            return
        
        self.running = True
        
        # ثبت callback برای پردازش پیام‌ها
        kafka_client.create_consumer(
            topic="transactions",
            group_id="fraud-detection-group",
            callback=self.process_transaction
        )
        logger.info("🚀 Fraud consumer service started")
    
    def stop(self):
        self.running = False
        logger.info("🛑 Fraud consumer service stopped")
        
        # بستن consumer اگر وجود داشته باشد
        if self.consumer:
            try:
                self.consumer.close()
                logger.info("✅ Consumer closed")

            except Exception as e:
                logger.error(f"❌ Error closing consumer: {e}")
            finally:
                self.consumer = None


# نمونه singleton
consumer_service = FraudConsumerService()