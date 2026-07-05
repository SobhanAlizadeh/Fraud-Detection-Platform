import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ml_engineering.training import ModelTrainer

def main():
    print("=" * 60)
    print("🚀 Fraud Detection Model Training")
    print("=" * 60)
    
    trainer = ModelTrainer()
    models, results = trainer.train_models()
    
    print("\n✅ Training completed successfully!")
    print(f"📊 Best model: {trainer.best_model_name}")
    print(f"📈 F1 Score: {results[trainer.best_model_name]['f1_score']:.4f}")

if __name__ == "__main__":
    main()
