import os
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

@dataclass
class DatabaseConfig:
    host: str = os.getenv("DB_HOST", "localhost")
    port: int = int(os.getenv("DB_PORT", "5432"))
    database: str = os.getenv("DB_NAME", "fraud_db")
    user: str = os.getenv("DB_USER", "fraud_user")
    password: str = os.getenv("DB_PASSWORD", "fraud_pass")
    
    @property
    def connection_string(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"

@dataclass
class KafkaConfig:
    bootstrap_servers: str = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    topic_transactions: str = "transactions"
    topic_predictions: str = "predictions"
    consumer_group: str = "fraud_detection_group"

@dataclass
class MLConfig:
    model_dir: Path = Path("models")
    feature_store_path: Path = Path("data/features")
    random_state: int = 42
    test_size: float = 0.2
    
@dataclass
class APIConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False

class Config:
    def __init__(self):
        self.db = DatabaseConfig()
        self.kafka = KafkaConfig()
        self.ml = MLConfig()
        self.api = APIConfig()
        
        # Create directories
        self.ml.model_dir.mkdir(exist_ok=True, parents=True)
        self.ml.feature_store_path.mkdir(exist_ok=True, parents=True)

config = Config()