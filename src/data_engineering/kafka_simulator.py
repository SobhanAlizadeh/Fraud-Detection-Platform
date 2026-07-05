"""
شبیه‌سازی تولید و مصرف داده در Kafka
برای تست سیستم بدون نیاز به Kafka واقعی
"""

import json
import random
import time
import threading
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
from queue import Queue, Empty
from dataclasses import dataclass, asdict
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============== مدل‌های داده ==============

@dataclass
class TransactionEvent:
    """مدل رویداد تراکنش"""
    transaction_id: str
    user_id: str
    amount: float
    timestamp: datetime
    merchant: str
    location: str
    transaction_type: str
    device_id: str
    ip_address: str
    is_fraud: bool = False
    fraud_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """تبدیل به دیکشنری"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TransactionEvent':
        """ساخت از دیکشنری"""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)

# ============== تولیدکننده داده ==============

class TransactionGenerator:
    """
    تولیدکننده تراکنش‌های شبیه‌سازی شده
    """
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)
        
        # لیست داده‌های نمونه
        self.merchants = [
            "Amazon", "Apple Store", "Google Play", "Spotify", "Netflix",
            "Uber", "DoorDash", "Walmart", "Target", "Best Buy",
            "Sephora", "Nike", "Adidas", "Zara", "H&M",
            "Starbucks", "McDonald's", "KFC", "Pizza Hut", "Domino's",
            "Cryptocurrency Exchange", "Online Casino", "Gambling Site",
            "Foreign Exchange", "Money Transfer"
        ]
        
        self.locations = [
            "Tehran", "Mashhad", "Isfahan", "Shiraz", "Tabriz",
            "Qom", "Karaj", "Ahvaz", "Rasht", "Kermanshah",
            "Zanjan", "Hamedan", "Yazd", "Ardabil", "Bandar Abbas"
        ]
        
        self.transaction_types = [
            "online", "in_store", "atm", "transfer", "payment"
        ]
        
        self.devices = [
            f"DEV-{i:04d}" for i in range(1, 101)
        ]
        
        self.ip_prefixes = [
            "192.168.1", "10.0.0", "172.16.0", "192.168.0",
            "10.1.1", "172.31.0", "192.168.100"
        ]
        
        # کاربران با پروفایل‌های مختلف ریسک
        self.user_profiles = self._create_user_profiles(50)
        
    def _create_user_profiles(self, num_users: int) -> Dict[str, Dict]:
        """ایجاد پروفایل‌های کاربری با ریسک‌های متفاوت"""
        profiles = {}
        
        for i in range(1, num_users + 1):
            user_id = f"user_{i:04d}"
            
            # تعیین سطح ریسک کاربر
            risk_level = random.choices(
                ['low', 'medium', 'high'],
                weights=[0.6, 0.3, 0.1]
            )[0]
            
            # پارامترهای تراکنش بر اساس سطح ریسک
            if risk_level == 'low':
                avg_amount = random.uniform(50, 500)
                amount_std = avg_amount * 0.3
                fraud_probability = 0.01
                transaction_frequency = random.randint(1, 10)  # تراکنش در روز
                
            elif risk_level == 'medium':
                avg_amount = random.uniform(100, 2000)
                amount_std = avg_amount * 0.5
                fraud_probability = 0.05
                transaction_frequency = random.randint(5, 20)
                
            else:  # high risk
                avg_amount = random.uniform(500, 10000)
                amount_std = avg_amount * 0.8
                fraud_probability = 0.15
                transaction_frequency = random.randint(10, 50)
            
            profiles[user_id] = {
                'risk_level': risk_level,
                'avg_amount': avg_amount,
                'amount_std': amount_std,
                'fraud_probability': fraud_probability,
                'transaction_frequency': transaction_frequency,
                'preferred_merchants': random.sample(self.merchants, k=random.randint(3, 8)),
                'preferred_locations': random.sample(self.locations, k=random.randint(1, 3)),
            }
            
        return profiles
    
    def generate_transaction(self, user_id: Optional[str] = None) -> TransactionEvent:
        """تولید یک تراکنش تصادفی"""
        
        # انتخاب کاربر
        if user_id is None:
            user_id = random.choice(list(self.user_profiles.keys()))
        
        profile = self.user_profiles[user_id]
        
        # تولید مبلغ بر اساس پروفایل کاربر
        amount = abs(random.gauss(profile['avg_amount'], profile['amount_std']))
        amount = round(amount, 2)
        
        # تعیین تقلبی بودن تراکنش
        is_fraud = random.random() < profile['fraud_probability']
        
        # اگر تقلبی باشد، مبلغ را غیرعادی می‌کنیم
        if is_fraud:
            amount = amount * random.uniform(2, 5)  # مبلغ بسیار بیشتر
            # یا تراکنش در مکان غیرمعمول
            location = random.choice(self.locations)
        else:
            location = random.choice(profile['preferred_locations'])
        
        # تولید زمان (در 24 ساعت اخیر)
        hours_ago = random.uniform(0, 24)
        timestamp = datetime.utcnow() - timedelta(hours=hours_ago)
        
        # انتخاب فروشنده
        if is_fraud:
            # فروشنده‌های پرریسک
            high_risk_merchants = [
                "Cryptocurrency Exchange", "Online Casino", 
                "Gambling Site", "Foreign Exchange", "Money Transfer"
            ]
            merchant = random.choice(high_risk_merchants)
        else:
            merchant = random.choice(profile['preferred_merchants'])
        
        # تولید شناسه یکتا
        transaction_id = f"TX{timestamp.strftime('%Y%m%d%H%M%S')}{random.randint(100, 999)}"
        
        return TransactionEvent(
            transaction_id=transaction_id,
            user_id=user_id,
            amount=amount,
            timestamp=timestamp,
            merchant=merchant,
            location=location,
            transaction_type=random.choice(self.transaction_types),
            device_id=random.choice(self.devices),
            ip_address=f"{random.choice(self.ip_prefixes)}.{random.randint(1, 254)}",
            is_fraud=is_fraud,
            fraud_score=1.0 if is_fraud else random.uniform(0, 0.3)
        )
    
    def generate_batch(self, size: int = 100) -> List[TransactionEvent]:
        """تولید دسته‌ای تراکنش"""
        return [self.generate_transaction() for _ in range(size)]
    
    def generate_stream(self, duration_seconds: int = 60, rate_per_second: float = 10):
        """تولید استریم تراکنش در زمان مشخص"""
        
        start_time = time.time()
        total_transactions = 0
        
        while time.time() - start_time < duration_seconds:
            # تولید تراکنش با نرخ مشخص
            batch_size = random.randint(1, int(rate_per_second * 1.5))
            transactions = self.generate_batch(batch_size)
            
            for transaction in transactions:
                yield transaction
                total_transactions += 1
            
            # منتظر ماندن برای حفظ نرخ
            time.sleep(1 / rate_per_second)
        
        logger.info(f"Generated {total_transactions} transactions in {duration_seconds}s")

# ============== شبیه‌ساز Kafka ==============

class KafkaSimulator:
    """
    شبیه‌ساز Kafka برای تست بدون نیاز به Kafka واقعی
    """
    
    def __init__(self):
        self.topics: Dict[str, Queue] = {}
        self.consumers: Dict[str, List[Callable]] = {}
        self.running = False
        self.threads: List[threading.Thread] = []
        self._lock = threading.Lock()
        
        logger.info("✅ Kafka Simulator initialized")
    
    def create_topic(self, topic_name: str, max_size: int = 1000):
        """ایجاد تاپیک جدید"""
        with self._lock:
            if topic_name not in self.topics:
                self.topics[topic_name] = Queue(maxsize=max_size)
                self.consumers[topic_name] = []
                logger.info(f"📝 Topic created: {topic_name}")
            else:
                logger.warning(f"⚠️ Topic {topic_name} already exists")
    
    def publish(self, topic_name: str, message: Dict[str, Any], key: Optional[str] = None):
        """انتشار پیام در تاپیک"""
        if topic_name not in self.topics:
            self.create_topic(topic_name)
        
        # افزودن timestamp و metadata
        if key:
            message['_key'] = key
        message['_timestamp'] = datetime.utcnow().isoformat()
        
        try:
            self.topics[topic_name].put_nowait(message)
            logger.debug(f"📤 Published to {topic_name}: {message.get('transaction_id', 'unknown')}")
            
            # اطلاع‌رسانی به مصرف‌کنندگان
            self._notify_consumers(topic_name, message)
            
        except Exception as e:
            logger.error(f"❌ Error publishing to {topic_name}: {e}")
    
    def subscribe(self, topic_name: str, callback: Callable):
        """ثبت مصرف‌کننده جدید"""
        if topic_name not in self.topics:
            self.create_topic(topic_name)
        
        with self._lock:
            if callback not in self.consumers[topic_name]:
                self.consumers[topic_name].append(callback)
                logger.info(f"👤 Consumer subscribed to {topic_name}")
            else:
                logger.warning(f"⚠️ Consumer already subscribed to {topic_name}")
    
    def _notify_consumers(self, topic_name: str, message: Dict[str, Any]):
        """اطلاع‌رسانی به تمام مصرف‌کنندگان"""
        for callback in self.consumers.get(topic_name, []):
            try:
                callback(message)
            except Exception as e:
                logger.error(f"❌ Consumer error: {e}")
    
    def start_consumer(self, topic_name: str, callback: Callable, 
                      poll_interval: float = 0.1):
        """شروع مصرف‌کننده در یک ترد جداگانه"""
        
        def consumer_loop():
            while self.running:
                try:
                    if topic_name in self.topics:
                        message = self.topics[topic_name].get(timeout=poll_interval)
                        callback(message)
                except Empty:
                    continue
                except Exception as e:
                    logger.error(f"❌ Consumer loop error: {e}")
                    time.sleep(1)
        
        thread = threading.Thread(
            target=consumer_loop,
            daemon=True,
            name=f"Consumer-{topic_name}"
        )
        self.threads.append(thread)
        thread.start()
        logger.info(f"🔄 Consumer thread started for {topic_name}")
    
    def start(self):
        """شروع شبیه‌ساز"""
        self.running = True
        logger.info("🚀 Kafka Simulator started")
    
    def stop(self):
        """توقف شبیه‌ساز"""
        self.running = False
        for thread in self.threads:
            thread.join(timeout=2)
        self.threads.clear()
        logger.info("🛑 Kafka Simulator stopped")
    
    def get_topic_size(self, topic_name: str) -> int:
        """دریافت تعداد پیام‌های موجود در تاپیک"""
        if topic_name in self.topics:
            return self.topics[topic_name].qsize()
        return 0
    
    def clear_topic(self, topic_name: str):
        """پاک کردن تاپیک"""
        if topic_name in self.topics:
            while not self.topics[topic_name].empty():
                try:
                    self.topics[topic_name].get_nowait()
                except Empty:
                    break
            logger.info(f"🧹 Topic cleared: {topic_name}")

# ============== سرویس تولیدکننده استریم ==============

class StreamProducerService:
    """
    سرویس تولید استریم داده به صورت مداوم
    """
    
    def __init__(self, kafka_simulator: KafkaSimulator, 
                 generator: TransactionGenerator):
        self.kafka = kafka_simulator
        self.generator = generator
        self.running = False
        self.producer_thread: Optional[threading.Thread] = None
        
    def start_producing(self, topic_name: str = "transactions", 
                       rate_per_second: float = 5,
                       duration: Optional[int] = None):
        """
        شروع تولید مداوم تراکنش
        
        Args:
            topic_name: نام تاپیک
            rate_per_second: نرخ تولید در ثانیه
            duration: مدت زمان تولید (ثانیه) - None برای بی‌نهایت
        """
        
        def producer_loop():
            start_time = time.time()
            count = 0
            
            while self.running:
                try:
                    # تولید تراکنش
                    transaction = self.generator.generate_transaction()
                    
                    # انتشار در Kafka
                    self.kafka.publish(
                        topic_name=topic_name,
                        message=transaction.to_dict(),
                        key=transaction.user_id
                    )
                    
                    count += 1
                    
                    # لاگ هر 10 تراکنش
                    if count % 10 == 0:
                        logger.info(f"📊 Produced {count} transactions to {topic_name}")
                    
                    # بررسی زمان
                    if duration:
                        elapsed = time.time() - start_time
                        if elapsed >= duration:
                            logger.info(f"⏰ Production duration ({duration}s) completed")
                            break
                    
                    # کنترل نرخ
                    time.sleep(1 / rate_per_second)
                    
                except Exception as e:
                    logger.error(f"❌ Producer error: {e}")
                    time.sleep(1)
            
            logger.info(f"🛑 Producer stopped. Total: {count} transactions")
        
        self.running = True
        self.producer_thread = threading.Thread(
            target=producer_loop,
            daemon=True,
            name="StreamProducer"
        )
        self.producer_thread.start()
        logger.info(f"🚀 Stream producer started on topic '{topic_name}' at {rate_per_second}/s")
        
        return self.producer_thread
    
    def stop_producing(self):
        """توقف تولید"""
        self.running = False
        if self.producer_thread:
            self.producer_thread.join(timeout=5)
        logger.info("🛑 Stream producer stopped")

# ============== سرویس مصرف‌کننده ==============

class StreamConsumerService:
    """
    سرویس مصرف و پردازش تراکنش‌ها
    """
    
    def __init__(self, kafka_simulator: KafkaSimulator):
        self.kafka = kafka_simulator
        self.transactions_processed = 0
        self.fraud_detected = 0
        self.processing_time_total = 0
        
    def process_transaction(self, message: Dict[str, Any]):
        """
        پردازش یک تراکنش دریافتی
        """
        start_time = time.time()
        
        try:
            # تبدیل به TransactionEvent
            transaction = TransactionEvent.from_dict(message)
            
            # پردازش (اینجا می‌توانیم مدل را صدا بزنیم)
            self.transactions_processed += 1
            
            # شبیه‌سازی تشخیص تقلب
            if transaction.is_fraud:
                self.fraud_detected += 1
                logger.info(f"🚨 FRAUD DETECTED: {transaction.transaction_id} - ${transaction.amount}")
            
            # لاگ هر 50 تراکنش
            if self.transactions_processed % 50 == 0:
                logger.info(f"📊 Processed {self.transactions_processed} transactions, "
                          f"Fraud: {self.fraud_detected} ({self.fraud_detected/self.transactions_processed*100:.2f}%)")
            
            # محاسبه زمان پردازش
            processing_time = time.time() - start_time
            self.processing_time_total += processing_time
            
        except Exception as e:
            logger.error(f"❌ Error processing transaction: {e}")
    
    def start_consuming(self, topic_name: str = "transactions"):
        """
        شروع مصرف تراکنش‌ها
        """
        self.kafka.subscribe(topic_name, self.process_transaction)
        self.kafka.start_consumer(topic_name, self.process_transaction)
        
        logger.info(f"🔄 Consumer started on topic '{topic_name}'")
        
    def get_stats(self) -> Dict[str, Any]:
        """
        دریافت آمار مصرف‌کننده
        """
        avg_processing_time = (
            self.processing_time_total / self.transactions_processed 
            if self.transactions_processed > 0 else 0
        )
        
        return {
            'total_processed': self.transactions_processed,
            'fraud_detected': self.fraud_detected,
            'fraud_rate': (
                self.fraud_detected / self.transactions_processed * 100
                if self.transactions_processed > 0 else 0
            ),
            'avg_processing_time_ms': avg_processing_time * 1000,
            'timestamp': datetime.utcnow().isoformat()
        }

# ============== مثال استفاده ==============

def main():
    """
    مثال اجرای شبیه‌ساز
    """
    logger.info("=" * 60)
    logger.info("🔄 Starting Kafka Simulator Demo")
    logger.info("=" * 60)
    
    # 1. ایجاد شبیه‌ساز
    kafka_sim = KafkaSimulator()
    kafka_sim.start()
    
    # 2. ایجاد تاپیک
    kafka_sim.create_topic("transactions")
    
    # 3. تولیدکننده
    generator = TransactionGenerator(seed=42)
    producer = StreamProducerService(kafka_sim, generator)
    
    # 4. مصرف‌کننده
    consumer = StreamConsumerService(kafka_sim)
    consumer.start_consuming("transactions")
    
    # 5. شروع تولید به مدت 10 ثانیه
    producer_thread = producer.start_producing(
        topic_name="transactions",
        rate_per_second=3,
        duration=10
    )
    
    # 6. منتظر ماندن برای اتمام تولید
    if producer_thread:
        producer_thread.join()
    
    # 7. آمار نهایی
    stats = consumer.get_stats()
    logger.info("\n" + "=" * 60)
    logger.info("📊 Final Statistics:")
    logger.info(f"   Total Transactions Processed: {stats['total_processed']}")
    logger.info(f"   Fraud Detected: {stats['fraud_detected']}")
    logger.info(f"   Fraud Rate: {stats['fraud_rate']:.2f}%")
    logger.info(f"   Avg Processing Time: {stats['avg_processing_time_ms']:.2f}ms")
    logger.info("=" * 60)
    
    # 8. توقف
    kafka_sim.stop()
    logger.info("✅ Demo completed")

if __name__ == "__main__":
    main()