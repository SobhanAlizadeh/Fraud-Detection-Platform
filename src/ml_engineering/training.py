# src/ml_engineering/trainer.py
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import joblib
from datetime import datetime
from typing import Dict, Any, Tuple, Optional, List
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, 
    roc_auc_score, 
    f1_score, 
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_curve
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

import mlflow
import mlflow.sklearn
import optuna

import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    کلاس آموزش مدل‌های مختلف برای تشخیص تقلب
    
    این کلاس شامل:
    - بارگذاری و پیش‌پردازش داده
    - آموزش مدل‌های XGBoost، LightGBM و CatBoost
    - بهینه‌سازی هایپرپارامترها با Optuna
    - ارزیابی و مقایسه مدل‌ها
    - ذخیره‌سازی بهترین مدل
    """
    
    def __init__(
        self, 
        data_path: Optional[Path] = None, 
        model_dir: Optional[Path] = None,
        random_state: int = 42
    ):
        """
        مقداردهی اولیه
        
        Args:
            data_path: مسیر دیتای پردازش شده
            model_dir: مسیر ذخیره مدل‌ها
            random_state: seed برای reproducible results
        """
        self.data_path = data_path or Path("data/processed")
        self.model_dir = model_dir or Path("models")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.random_state = random_state
        
        # پیش‌پردازشگرها
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
        # ذخیره بهترین مدل
        self.best_model = None
        self.best_model_name = None
        self.best_metrics = {}
        
        # دایرکتوری گزارش‌ها
        self.reports_dir = Path("reports")
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        # دایرکتوری نمودارها
        self.plots_dir = self.reports_dir / "plots"
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("=" * 60)
        logger.info("✅ ModelTrainer Initialized")
        logger.info(f"   Data path: {self.data_path}")
        logger.info(f"   Model dir: {self.model_dir}")
        logger.info(f"   Reports dir: {self.reports_dir}")
        logger.info("=" * 60)
    
    # ============================================================
    # 1. بارگذاری داده
    # ============================================================
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        بارگذاری داده از فایل یا تولید داده نمونه
        
        Returns:
            X: ویژگی‌ها (DataFrame)
            y: برچسب‌ها (Series)
        """
        try:
            # تلاش برای بارگذاری از فایل
            data_file = self.data_path / "features.parquet"
            
            if not data_file.exists():
                logger.warning(f"⚠️ Data file not found: {data_file}")
                logger.info("📊 Generating sample data for testing...")
                return self._generate_sample_data()
            
            df = pd.read_parquet(data_file)
            logger.info(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
            
            # شناسایی ستون هدف
            target_cols = ['is_fraud', 'fraud', 'target', 'label']
            target = None
            
            for col in target_cols:
                if col in df.columns:
                    target = col
                    break
            
            if target is None:
                raise ValueError("Target column not found in data")
            
            # جدا کردن ویژگی‌ها و هدف
            X = df.drop(columns=[target])
            y = df[target]
            
            # نمایش آمار اولیه
            fraud_rate = y.sum() / len(y) * 100
            logger.info(f"📊 Fraud rate: {fraud_rate:.2f}%")
            logger.info(f"   Features: {X.shape[1]}, Samples: {X.shape[0]}")
            
            return X, y
            
        except Exception as e:
            logger.error(f"❌ Error loading data: {e}")
            logger.info("📊 Generating sample data for testing...")
            return self._generate_sample_data()
    
    def _generate_sample_data(self, n_samples: int = 10000) -> Tuple[pd.DataFrame, pd.Series]:
        """
        تولید داده‌های نمونه برای تست
        
        Args:
            n_samples: تعداد نمونه‌ها
            
        Returns:
            X: ویژگی‌ها
            y: برچسب‌ها
        """
        np.random.seed(self.random_state)
        
        # تولید ویژگی‌های تصادفی
        n_features = 15
        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        
        # ایجاد برچسب با نرخ تقلب ۸٪
        y = pd.Series(
            np.random.choice([0, 1], n_samples, p=[0.92, 0.08]),
            name='is_fraud'
        )
        
        # افزودن روابط بین ویژگی‌ها و برچسب
        X['feature_0'] = X['feature_0'] + 0.8 * y + np.random.randn(n_samples) * 0.1
        X['feature_1'] = X['feature_1'] - 0.5 * y + np.random.randn(n_samples) * 0.1
        X['feature_2'] = X['feature_2'] + 0.3 * y * X['feature_3']
        
        logger.info(f"✅ Generated {n_samples} sample records with {X.shape[1]} features")
        logger.info(f"   Fraud rate: {y.sum()/len(y)*100:.2f}%")
        
        return X, y
    
    # ============================================================
    # 2. پیش‌پردازش داده
    # ============================================================
    
    def preprocess_data(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        is_training: bool = True
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        پیش‌پردازش داده‌ها:
        - کد کردن ستون‌های کتگوری
        - اسکیل کردن ویژگی‌های عددی
        
        Args:
            X: ویژگی‌ها
            y: برچسب‌ها
            is_training: آیا در حالت آموزش هستیم؟
            
        Returns:
            X_processed: داده‌های پردازش شده
            y: برچسب‌ها
        """
        X_processed = X.copy()
        
        # 1. شناسایی ستون‌های کتگوری
        categorical_cols = X_processed.select_dtypes(include=['object', 'category']).columns
        categorical_cols = [col for col in categorical_cols if col not in ['transaction_id', 'user_id']]
        
        if len(categorical_cols) > 0:
            logger.info(f"📋 Categorical columns: {list(categorical_cols)}")
        
        # 2. کد کردن ستون‌های کتگوری
        for col in categorical_cols:
            if is_training:
                self.label_encoders[col] = LabelEncoder()
                X_processed[col] = self.label_encoders[col].fit_transform(
                    X_processed[col].astype(str)
                )
            else:
                if col in self.label_encoders:
                    X_processed[col] = X_processed[col].astype(str).map(
                        lambda x: self.label_encoders[col].transform([x])[0]
                        if x in self.label_encoders[col].classes_
                        else -1
                    )
                else:
                    # اگر encoder وجود نداشت، ستون را حذف می‌کنیم
                    X_processed = X_processed.drop(columns=[col])
                    logger.warning(f"⚠️ Column '{col}' has no encoder, dropping it")
        
        # 3. شناسایی ستون‌های عددی
        numeric_cols = X_processed.select_dtypes(include=[np.number]).columns
        
        # 4. اسکیل کردن ویژگی‌های عددی
        if len(numeric_cols) > 0:
            if is_training:
                X_scaled = self.scaler.fit_transform(X_processed[numeric_cols])
            else:
                X_scaled = self.scaler.transform(X_processed[numeric_cols])
            
            X_processed[numeric_cols] = X_scaled
        
        logger.info(f"✅ Data preprocessed: {X_processed.shape[1]} features")
        return X_processed, y
    
    # ============================================================
    # 3. بالانس کردن داده
    # ============================================================
    
    def balance_data(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        method: str = 'smote'
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        بالانس کردن داده‌ها با روش‌های مختلف
        
        Args:
            X: ویژگی‌ها
            y: برچسب‌ها
            method: روش بالانس ('smote', 'random_oversample', 'random_undersample')
            
        Returns:
            X_balanced: ویژگی‌های بالانس شده
            y_balanced: برچسب‌های بالانس شده
        """
        if method == 'smote':
            smote = SMOTE(random_state=self.random_state)
            X_balanced, y_balanced = smote.fit_resample(X, y)
            
        else:
            # روش‌های دیگر می‌توانند اضافه شوند
            logger.warning(f"⚠️ Method '{method}' not implemented, using SMOTE")
            smote = SMOTE(random_state=self.random_state)
            X_balanced, y_balanced = smote.fit_resample(X, y)
        
        logger.info(f"✅ Data balanced: {y_balanced.sum()} fraud samples out of {len(y_balanced)}")
        logger.info(f"   Fraud rate: {y_balanced.sum()/len(y_balanced)*100:.2f}%")
        
        return X_balanced, y_balanced
    
    # ============================================================
    # 4. بهینه‌سازی هایپرپارامترها
    # ============================================================
    
    def optimize_hyperparameters(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series, 
        model_type: str = 'xgboost', 
        n_trials: int = 30
    ) -> Dict[str, Any]:
        """
        بهینه‌سازی هایپرپارامترها با Optuna
        
        Args:
            X_train: داده‌های آموزش
            y_train: برچسب‌های آموزش
            model_type: نوع مدل ('xgboost', 'lightgbm', 'catboost')
            n_trials: تعداد دفعات بهینه‌سازی
            
        Returns:
            بهترین پارامترها
        """
        logger.info(f"🔍 Optimizing {model_type} hyperparameters with {n_trials} trials...")
        
        def objective(trial):
            if model_type == 'xgboost':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 500, step=50),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                    'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1, 20),
                    'gamma': trial.suggest_float('gamma', 0, 0.5),
                    'eval_metric': 'logloss',
                    'verbosity': 0,
                    'random_state': self.random_state,
                }
                if 'use_label_encoder' in params:
                    del params['use_label_encoder']
                model = xgb.XGBClassifier(**params)
                
            elif model_type == 'lightgbm':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 500, step=50),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'num_leaves': trial.suggest_int('num_leaves', 10, 100),
                    'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
                    'min_child_samples': trial.suggest_int('min_child_samples', 5, 30),
                    'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1, 20),
                    'verbosity': -1,
                    'random_state': self.random_state,
                }
                model = lgb.LGBMClassifier(**params)
                
            else:  # catboost
                params = {
                    'iterations': trial.suggest_int('iterations', 100, 500, step=50),
                    'depth': trial.suggest_int('depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
                    'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1, 20),
                    'eval_metric': 'Logloss',
                    'verbose': False,
                    'random_state': self.random_state,
                }
                model = CatBoostClassifier(**params)
            
            # Cross-validation
            scores = cross_val_score(model, X_train, y_train, cv=3, scoring='f1')
            return scores.mean()
        
        # ایجاد و اجرای مطالعه
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        
        logger.info(f"✅ Best {model_type} parameters: {study.best_params}")
        logger.info(f"   Best F1 score (CV): {study.best_value:.4f}")
        
        return study.best_params
    
    # ============================================================
    # 5. آموزش مدل
    # ============================================================
    
    def train_single_model(
        self, 
        model_name: str, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        X_test: pd.DataFrame, 
        y_test: pd.Series,
        optimize: bool = True
    ) -> Tuple[Any, Dict[str, float]]:
        """
        آموزش یک مدل
        
        Args:
            model_name: نام مدل ('xgboost', 'lightgbm', 'catboost')
            X_train: داده‌های آموزش
            y_train: برچسب‌های آموزش
            X_test: داده‌های تست
            y_test: برچسب‌های تست
            optimize: آیا هایپرپارامترها بهینه شوند؟
            
        Returns:
            مدل آموزش دیده و متریک‌ها
        """
        logger.info("=" * 50)
        logger.info(f"🚀 Training {model_name}...")
        
        # بهینه‌سازی هایپرپارامترها
        if optimize:
            best_params = self.optimize_hyperparameters(
                X_train, y_train, model_name, n_trials=20
            )
        else:
            # پارامترهای پیش‌فرض
            best_params = {}
            if model_name == 'xgboost':
                best_params = {
                    'n_estimators': 200,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'scale_pos_weight': 5,
                    'random_state': self.random_state,
                    'verbosity': 0,
                }
            elif model_name == 'lightgbm':
                best_params = {
                    'n_estimators': 200,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'scale_pos_weight': 5,
                    'random_state': self.random_state,
                    'verbosity': -1,
                }
            else:
                best_params = {
                    'iterations': 200,
                    'depth': 6,
                    'learning_rate': 0.1,
                    'scale_pos_weight': 5,
                    'random_state': self.random_state,
                    'verbose': False,
                }
        
        # ایجاد مدل
        if model_name == 'xgboost':
            model = xgb.XGBClassifier(**best_params)
        elif model_name == 'lightgbm':
            model = lgb.LGBMClassifier(**best_params)
        else:
            model = CatBoostClassifier(**best_params)
        
        # آموزش
        model.fit(X_train, y_train)
        
        # پیش‌بینی
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        # محاسبه متریک‌ها
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_proba),
        }
        
        # نمایش نتایج
        logger.info(f"✅ {model_name} trained successfully!")
        logger.info(f"   Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"   Precision: {metrics['precision']:.4f}")
        logger.info(f"   Recall: {metrics['recall']:.4f}")
        logger.info(f"   F1 Score: {metrics['f1_score']:.4f}")
        logger.info(f"   ROC-AUC: {metrics['roc_auc']:.4f}")
        
        # ذخیره مدل
        model_path = self.model_dir / f"{model_name}_model.pkl"
        joblib.dump(model, model_path)
        logger.info(f"💾 Model saved: {model_path}")
        
        return model, metrics
    
    # ============================================================
    # 6. آموزش همه مدل‌ها
    # ============================================================
    
    def train_models(
        self, 
        optimize: bool = True,
        save_reports: bool = True
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        آموزش تمام مدل‌ها
        
        Args:
            optimize: آیا هایپرپارامترها بهینه شوند؟
            save_reports: آیا گزارش‌ها ذخیره شوند؟
            
        Returns:
            models: دیکشنری مدل‌ها
            results: دیکشنری نتایج
        """
        logger.info("=" * 60)
        logger.info("🚀 Starting Model Training Pipeline")
        logger.info("=" * 60)
        
        # 1. بارگذاری داده
        X, y = self.load_data()
        
        # 2. تقسیم داده‌ها
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        logger.info(f"📊 Train size: {len(X_train)}, Test size: {len(X_test)}")
        
        # 3. پیش‌پردازش
        X_train_processed, y_train_processed = self.preprocess_data(
            X_train, y_train, is_training=True
        )
        X_test_processed, y_test_processed = self.preprocess_data(
            X_test, y_test, is_training=False
        )
        
        # 4. بالانس کردن
        X_train_balanced, y_train_balanced = self.balance_data(
            X_train_processed, y_train_processed
        )
        
        # 5. تنظیم MLflow
        mlflow.set_experiment("Fraud Detection")
        
        models = {}
        results = {}
        
        # 6. آموزش مدل‌ها
        model_names = ['xgboost', 'lightgbm', 'catboost']
        
        for model_name in model_names:
            with mlflow.start_run(run_name=model_name):
                # آموزش
                model, metrics = self.train_single_model(
                    model_name, 
                    X_train_balanced, y_train_balanced,
                    X_test_processed, y_test_processed,
                    optimize=optimize
                )
                
                models[model_name] = model
                results[model_name] = metrics
                
                # لاگ در MLflow
                mlflow.log_params(
                    self.optimize_hyperparameters(
                        X_train_balanced, y_train_balanced, model_name, n_trials=5
                    )
                )
                for key, value in metrics.items():
                    mlflow.log_metric(key, value)
                
                mlflow.sklearn.log_model(model, model_name)
        
        # 7. انتخاب بهترین مدل
        best_model_name = max(results, key=lambda x: results[x]['f1_score'])
        self.best_model = models[best_model_name]
        self.best_model_name = best_model_name
        self.best_metrics = results[best_model_name]
        
        # 8. ذخیره بهترین مدل
        best_model_path = self.model_dir / "best_model.pkl"
        joblib.dump(self.best_model, best_model_path)
        logger.info(f"💾 Best model saved: {best_model_path}")
        
        # 9. نمایش نتایج نهایی
        logger.info("=" * 60)
        logger.info("🏆 Training Complete!")
        logger.info(f"   Best Model: {best_model_name}")
        logger.info(f"   F1 Score: {results[best_model_name]['f1_score']:.4f}")
        logger.info(f"   ROC-AUC: {results[best_model_name]['roc_auc']:.4f}")
        logger.info("=" * 60)
        
        # 10. ذخیره گزارش‌ها
        if save_reports:
            self._save_reports(results)
            self._plot_results(results, X_test, y_test)
        
        return models, results
    
    # ============================================================
    # 7. ذخیره گزارش‌ها و نمودارها
    # ============================================================
    
    def _save_reports(self, results: Dict[str, Dict[str, float]]):
        """
        ذخیره گزارش‌های نتایج
        
        Args:
            results: دیکشنری نتایج
        """
        import json
        
        report = {
            'best_model': self.best_model_name,
            'best_metrics': self.best_metrics,
            'all_models': results,
            'timestamp': datetime.now().isoformat(),
            'random_state': self.random_state,
            'data_path': str(self.data_path),
        }
        
        report_file = self.reports_dir / "training_results.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"📊 Report saved: {report_file}")
    
    def _plot_results(self, results: Dict[str, Dict[str, float]], X_test, y_test):
        """
        رسم نمودارهای مقایسه‌ای
        
        Args:
            results: دیکشنری نتایج
            X_test: داده‌های تست (برای اهمیت ویژگی‌ها)
            y_test: برچسب‌های تست
        """
        try:
            # 1. مقایسه متریک‌ها
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # نمودار میله‌ای متریک‌ها
            metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
            models = list(results.keys())
            
            x = np.arange(len(metrics_to_plot))
            width = 0.25
            
            for i, model in enumerate(models):
                values = [results[model].get(m, 0) for m in metrics_to_plot]
                axes[0].bar(x + i*width, values, width, label=model)
            
            axes[0].set_xlabel('Metrics')
            axes[0].set_ylabel('Score')
            axes[0].set_title('Model Comparison')
            axes[0].set_xticks(x + width)
            axes[0].set_xticklabels(metrics_to_plot)
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # 2. اهمیت ویژگی‌ها (برای XGBoost)
            if 'xgboost' in self.best_model:
                importances = self.best_model.feature_importances_
                feature_names = X_test.columns[:len(importances)]
                
                indices = np.argsort(importances)[-10:]
                
                axes[1].barh(range(len(indices)), importances[indices])
                axes[1].set_yticks(range(len(indices)))
                axes[1].set_yticklabels([feature_names[i] for i in indices])
                axes[1].set_xlabel('Feature Importance')
                axes[1].set_title('Top 10 Features (XGBoost)')
                axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.plots_dir / "model_comparison.png", dpi=150)
            plt.close()
            
            logger.info(f"📊 Plots saved: {self.plots_dir}")
            
        except Exception as e:
            logger.warning(f"⚠️ Could not create plots: {e}")


# ============================================================
# 8. اجرای مستقیم
# ============================================================

if __name__ == "__main__":
    # ایجاد نمونه و آموزش
    trainer = ModelTrainer()
    models, results = trainer.train_models(optimize=True)
    
    print("\n" + "=" * 60)
    print("✅ Training completed successfully!")
    print(f"🏆 Best model: {trainer.best_model_name}")
    print(f"📈 F1 Score: {trainer.best_metrics['f1_score']:.4f}")
    print(f"📈 ROC-AUC: {trainer.best_metrics['roc_auc']:.4f}")
    print("=" * 60)