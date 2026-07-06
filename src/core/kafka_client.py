import json
import logging
from datetime import datetime, date  # <--- این خط رو اضافه کنید
from typing import Dict, Any, Optional, Callable
from kafka import KafkaProducer, KafkaConsumer
from kafka.errors import KafkaError
import os
import threading
import time


logger = logging.getLogger(__name__)

class KafkaClient:
    def __init__(self):
        self.bootstrap_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
        self.producer = None
        self.consumer = None
        self._init_producer()
        
    def _init_producer(self):
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                key_serializer=lambda k: k.encode('utf-8') if k else None,
                acks='all',
                retries=3
            )
            logger.info(f"✅ Kafka producer connected to {self.bootstrap_servers}")
        except Exception as e:
            logger.error(f"❌ Failed to create Kafka producer: {e}")
            self.producer = None
    
    def publish(self, topic: str, message: Dict[str, Any], key: Optional[str] = None):
        if not self.producer:
            logger.error("❌ Kafka producer not available")
            return False
        try:
            # 1. کپی از پیام برای جلوگیری از تغییرات ناخواسته
            message_copy = message.copy()
            
            # 2. حلقه روی آیتم‌های پیام برای تبدیل تاریخ‌ها
            for k, v in message_copy.items():
                # بررسی درست نوع datetime
                if isinstance(v, (datetime, date)):  # <-- اینجا باید به این شکل باشد
                    message_copy[k] = v.isoformat()
                # اگر نوع دیکشنری یا لیست داشت، می‌توانید اینجا پردازش کنید
            
            # 3. ارسال به کافکا
            future = self.producer.send(topic, value=message_copy, key=key)
            result = future.get(timeout=5)
            logger.debug(f"📤 Published to {topic}: {message_copy.get('transaction_id', 'unknown')}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to publish to {topic}: {e}")
            return False
    
    def create_consumer(self, topic: str, group_id: str, callback: Callable):
        try:
            from kafka import KafkaConsumer
            
            consumer = KafkaConsumer(
                topic,
                bootstrap_servers=self.bootstrap_servers,
                group_id=group_id,
                value_deserializer=lambda v: json.loads(v.decode('utf-8')),
                auto_offset_reset='earliest',
                enable_auto_commit=True
            )
            
            logger.info(f"🔄 Kafka consumer started for topic '{topic}'")
            
            # مصرف‌کننده در یک ترد جداگانه اجرا میشه
            import threading
            def consume_loop():
                for message in consumer:
                    try:
                        callback(message.value)
                    except Exception as e:
                        logger.error(f"❌ Error processing message: {e}")
            
            thread = threading.Thread(target=consume_loop, daemon=True)
            thread.start()
            return consumer
            
        except Exception as e:
            logger.error(f"❌ Failed to create consumer: {e}")
            return None
    
    def close(self):
        if self.producer:
            self.producer.close()
            logger.info("✅ Kafka producer closed")

# نمونه Singleton
kafka_client = KafkaClient()