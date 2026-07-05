"""
مانیتورینگ و پایش مدل با استفاده از Evidently
بررسی Data Drift, Target Drift, و Performance
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import json
import logging
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Evidently imports
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import (
    DataDriftPreset,
    TargetDriftPreset,
    DataQualityPreset,
    ClassificationPreset
)
from evidently.metrics import (
    ColumnCorrelationsMetric,
    ColumnDistributionMetric,
    ColumnQuantileMetric,
    ColumnSummaryMetric,
    ColumnValuePlot,
    DatasetDriftMetric,
    DatasetMissingValuesMetric,
    DatasetSummaryMetric
)
from evidently.metrics.base_metric import Metric
from evidently.test_preset import DataStabilityTestPreset
from evidently.test_suite import TestSuite

from ..core.database import db_manager, Transaction, Prediction
from ..core.config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============== کلاس مدیریت مانیتورینگ ==============

class ModelMonitor:
    """
    کلاس اصلی برای مانیتورینگ مدل
    """
    
    def __init__(self, reference_data_path: Optional[Path] = None):
        self.reference_data = None
        self.column_mapping = None
        self.monitoring_dir = Path("monitoring_reports")
        self.monitoring_dir.mkdir(exist_ok=True)
        
        # بارگذاری دیتای مرجع
        if reference_data_path and reference_data_path.exists():
            self.reference_data = pd.read_parquet(reference_data_path)
            self._setup_column_mapping()
            logger.info(f"✅ Reference data loaded: {len(self.reference_data)} rows")
        else:
            logger.warning("⚠️ No reference data provided")
    
    def _setup_column_mapping(self):
        """تنظیم نقشه ستون‌ها برای Evidently"""
        if self.reference_data is not None:
            self.column_mapping = ColumnMapping()
            self.column_mapping.target = 'is_fraud'
            self.column_mapping.prediction = 'prediction'
            self.column_mapping.numerical_features = [
                'amount', 'transaction_hour', 'transaction_day_of_week',
                'avg_transaction_7d', 'transaction_frequency_7d',
                'amount_to_avg_ratio', 'amount_hour_interaction'
            ]
            self.column_mapping.categorical_features = [
                'merchant', 'location', 'transaction_type',
                'device_id', 'merchant_risk'
            ]
    
    def load_current_data(self, days: int = 7) -> pd.DataFrame:
        """
        بارگذاری داده‌های جاری از دیتابیس
        
        Args:
            days: تعداد روزهای گذشته
            
        Returns:
            DataFrame داده‌های جاری
        """
        session = db_manager.get_session()
        
        try:
            # دریافت تراکنش‌های اخیر
            start_date = datetime.utcnow() - timedelta(days=days)
            
            transactions = session.query(Transaction).filter(
                Transaction.timestamp >= start_date
            ).all()
            
            if not transactions:
                logger.warning("⚠️ No transactions found in the last {days} days")
                return pd.DataFrame()
            
            # تبدیل به DataFrame
            data = []
            for t in transactions:
                # دریافت پیش‌بینی مرتبط
                prediction = session.query(Prediction).filter(
                    Prediction.transaction_id == t.transaction_id
                ).first()
                
                row = {
                    'transaction_id': t.transaction_id,
                    'user_id': t.user_id,
                    'amount': t.amount,
                    'timestamp': t.timestamp,
                    'merchant': t.merchant,
                    'location': t.location,
                    'transaction_type': t.transaction_type,
                    'device_id': t.device_id,
                    'ip_address': t.ip_address,
                    'is_fraud': t.is_fraud,
                    'fraud_score': t.fraud_score
                }
                
                # افزودن فیچرها
                if t.features:
                    row.update(t.features)
                
                # افزودن پیش‌بینی
                if prediction:
                    row['prediction'] = prediction.is_fraud_predicted
                    row['fraud_probability'] = prediction.fraud_probability
                
                data.append(row)
            
            df = pd.DataFrame(data)
            
            # تبدیل timestamp
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # استخراج فیچرهای زمانی
            if not df.empty:
                df['transaction_hour'] = df['timestamp'].dt.hour
                df['transaction_day_of_week'] = df['timestamp'].dt.dayofweek
                df['transaction_date'] = df['timestamp'].dt.date
            
            logger.info(f"✅ Loaded {len(df)} transactions for monitoring")
            return df
            
        except Exception as e:
            logger.error(f"❌ Error loading current data: {e}")
            return pd.DataFrame()
        finally:
            session.close()
    
    # ============== گزارش‌های مختلف ==============
    
    def generate_data_drift_report(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """
        تولید گزارش Data Drift
        
        Args:
            current_data: داده‌های جاری
            
        Returns:
            دیکشنری شامل نتایج
        """
        if self.reference_data is None or current_data.empty:
            return {'error': 'Reference data or current data is empty'}
        
        try:
            # ایجاد گزارش
            report = Report(metrics=[
                DataDriftPreset(),
                DatasetSummaryMetric(),
                DatasetMissingValuesMetric(),
            ])
            
            # اجرای گزارش
            report.run(
                reference_data=self.reference_data,
                current_data=current_data,
                column_mapping=self.column_mapping
            )
            
            # ذخیره گزارش
            report_path = self.monitoring_dir / f"data_drift_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.html"
            report.save_html(str(report_path))
            
            # استخراج نتایج
            results = report.as_dict()
            
            # استخراج خلاصه
            summary = {
                'timestamp': datetime.utcnow().isoformat(),
                'reference_data_size': len(self.reference_data),
                'current_data_size': len(current_data),
                'drift_detected': results.get('metrics', {}).get('dataset_drift', {}).get('result', {}).get('drift_detected', False),
                'drift_score': results.get('metrics', {}).get('dataset_drift', {}).get('result', {}).get('drift_score', 0),
                'drifted_columns': results.get('metrics', {}).get('data_drift', {}).get('result', {}).get('drifted_columns', []),
                'report_path': str(report_path)
            }
            
            logger.info(f"📊 Data drift report generated: {summary['drift_detected']}")
            return summary
            
        except Exception as e:
            logger.error(f"❌ Error generating data drift report: {e}")
            return {'error': str(e)}
    
    def generate_performance_report(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """
        تولید گزارش عملکرد مدل
        
        Args:
            current_data: داده‌های جاری با مقادیر واقعی
            
        Returns:
            دیکشنری شامل نتایج
        """
        if current_data.empty:
            return {'error': 'Current data is empty'}
        
        try:
            # نیاز به ستون‌های target و prediction
            required_cols = ['is_fraud', 'prediction']
            if not all(col in current_data.columns for col in required_cols):
                logger.warning("⚠️ Missing target or prediction columns")
                return {'error': 'Missing required columns'}
            
            # ایجاد گزارش
            report = Report(metrics=[
                ClassificationPreset(),
                DatasetSummaryMetric(),
            ])
            
            # اگر دیتای مرجع داریم، از آن استفاده می‌کنیم
            if self.reference_data is not None:
                report.run(
                    reference_data=self.reference_data,
                    current_data=current_data,
                    column_mapping=self.column_mapping
                )
            else:
                report.run(
                    current_data=current_data,
                    column_mapping=self.column_mapping
                )
            
            # ذخیره گزارش
            report_path = self.monitoring_dir / f"performance_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.html"
            report.save_html(str(report_path))
            
            # استخراج نتایج
            results = report.as_dict()
            
            # استخراج متریک‌های کلیدی
            metrics = results.get('metrics', {})
            
            # محاسبه متریک‌های اصلی
            accuracy = metrics.get('accuracy', {}).get('result', {}).get('value')
            precision = metrics.get('precision', {}).get('result', {}).get('value')
            recall = metrics.get('recall', {}).get('result', {}).get('value')
            f1 = metrics.get('f1', {}).get('result', {}).get('value')
            roc_auc = metrics.get('roc_auc', {}).get('result', {}).get('value')
            
            summary = {
                'timestamp': datetime.utcnow().isoformat(),
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': roc_auc,
                'data_size': len(current_data),
                'report_path': str(report_path)
            }
            
            logger.info(f"📊 Performance report generated - F1: {f1:.4f}")
            return summary
            
        except Exception as e:
            logger.error(f"❌ Error generating performance report: {e}")
            return {'error': str(e)}
    
    def generate_target_drift_report(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """
        تولید گزارش Target Drift
        
        Args:
            current_data: داده‌های جاری
            
        Returns:
            دیکشنری شامل نتایج
        """
        if self.reference_data is None or current_data.empty:
            return {'error': 'Reference data or current data is empty'}
        
        try:
            # ایجاد گزارش
            report = Report(metrics=[
                TargetDriftPreset(),
                DatasetSummaryMetric(),
            ])
            
            # اجرای گزارش
            report.run(
                reference_data=self.reference_data,
                current_data=current_data,
                column_mapping=self.column_mapping
            )
            
            # ذخیره گزارش
            report_path = self.monitoring_dir / f"target_drift_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.html"
            report.save_html(str(report_path))
            
            summary = {
                'timestamp': datetime.utcnow().isoformat(),
                'drift_detected': True,  # از گزارش استخراج کن
                'report_path': str(report_path)
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"❌ Error generating target drift report: {e}")
            return {'error': str(e)}
    
    def generate_data_quality_report(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """
        تولید گزارش کیفیت داده
        
        Args:
            current_data: داده‌های جاری
            
        Returns:
            دیکشنری شامل نتایج
        """
        if current_data.empty:
            return {'error': 'Current data is empty'}
        
        try:
            # ایجاد گزارش
            report = Report(metrics=[
                DataQualityPreset(),
                DatasetMissingValuesMetric(),
                DatasetSummaryMetric(),
                ColumnSummaryMetric(column_name='amount'),
                ColumnDistributionMetric(column_name='amount'),
            ])
            
            # اجرای گزارش
            if self.reference_data is not None:
                report.run(
                    reference_data=self.reference_data,
                    current_data=current_data,
                    column_mapping=self.column_mapping
                )
            else:
                report.run(
                    current_data=current_data,
                    column_mapping=self.column_mapping
                )
            
            # ذخیره گزارش
            report_path = self.monitoring_dir / f"data_quality_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.html"
            report.save_html(str(report_path))
            
            summary = {
                'timestamp': datetime.utcnow().isoformat(),
                'missing_values': current_data.isnull().sum().to_dict(),
                'data_size': len(current_data),
                'report_path': str(report_path)
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"❌ Error generating data quality report: {e}")
            return {'error': str(e)}
    
    # ============== تست‌های خودکار ==============
    
    def run_monitoring_tests(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """
        اجرای تست‌های مانیتورینگ
        
        Args:
            current_data: داده‌های جاری
            
        Returns:
            دیکشنری شامل نتایج تست‌ها
        """
        if current_data.empty:
            return {'error': 'Current data is empty'}
        
        try:
            test_suite = TestSuite(tests=[
                DataStabilityTestPreset(),
            ])
            
            if self.reference_data is not None:
                test_suite.run(
                    reference_data=self.reference_data,
                    current_data=current_data,
                    column_mapping=self.column_mapping
                )
            else:
                test_suite.run(
                    current_data=current_data,
                    column_mapping=self.column_mapping
                )
            
            # ذخیره نتایج
            test_path = self.monitoring_dir / f"monitoring_tests_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.html"
            test_suite.save_html(str(test_path))
            
            # استخراج نتایج
            results = test_suite.as_dict()
            
            summary = {
                'timestamp': datetime.utcnow().isoformat(),
                'tests_passed': results.get('summary', {}).get('passed', 0),
                'tests_failed': results.get('summary', {}).get('failed', 0),
                'total_tests': results.get('summary', {}).get('total', 0),
                'success_rate': results.get('summary', {}).get('passed', 0) / max(1, results.get('summary', {}).get('total', 1)),
                'test_path': str(test_path)
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"❌ Error running monitoring tests: {e}")
            return {'error': str(e)}
    
    # ============== داشبورد بصری ==============
    
    def create_visualization_dashboard(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """
        ایجاد داشبورد بصری با Plotly
        
        Args:
            current_data: داده‌های جاری
            
        Returns:
            دیکشنری شامل JSON شکل‌ها
        """
        if current_data.empty:
            return {'error': 'Current data is empty'}
        
        figures = {}
        
        try:
            # 1. توزیع مبلغ تراکنش‌ها
            fig1 = go.Figure()
            fig1.add_trace(go.Histogram(
                x=current_data['amount'],
                nbinsx=50,
                name='Current Data',
                marker_color='blue'
            ))
            if self.reference_data is not None and 'amount' in self.reference_data.columns:
                fig1.add_trace(go.Histogram(
                    x=self.reference_data['amount'],
                    nbinsx=50,
                    name='Reference Data',
                    marker_color='red',
                    opacity=0.6
                ))
            fig1.update_layout(
                title='Distribution of Transaction Amounts',
                xaxis_title='Amount',
                yaxis_title='Frequency',
                barmode='overlay'
            )
            figures['amount_distribution'] = fig1.to_json()
            
            # 2. تراکنش‌های تقلبی در طول زمان
            if 'timestamp' in current_data.columns:
                daily_data = current_data.groupby(
                    current_data['timestamp'].dt.date
                ).agg({
                    'is_fraud': ['count', 'sum']
                }).reset_index()
                daily_data.columns = ['date', 'total', 'fraud']
                
                fig2 = make_subplots(specs=[[{"secondary_y": True}]])
                fig2.add_trace(
                    go.Bar(x=daily_data['date'], y=daily_data['total'], name='Total Transactions'),
                    secondary_y=False,
                )
                fig2.add_trace(
                    go.Scatter(x=daily_data['date'], y=daily_data['fraud'], name='Fraud Transactions', line=dict(color='red')),
                    secondary_y=True,
                )
                fig2.update_layout(
                    title='Transactions and Fraud Over Time',
                    xaxis_title='Date',
                    barmode='group'
                )
                fig2.update_yaxes(title_text="Total Transactions", secondary_y=False)
                fig2.update_yaxes(title_text="Fraud Transactions", secondary_y=True)
                figures['transactions_over_time'] = fig2.to_json()
            
            # 3. نقشه حرارتی همبستگی
            if len(current_data.select_dtypes(include=[np.number]).columns) > 1:
                numeric_cols = current_data.select_dtypes(include=[np.number]).columns
                corr_matrix = current_data[numeric_cols].corr()
                
                fig3 = go.Figure(data=go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.columns,
                    colorscale='RdBu',
                    zmin=-1,
                    zmax=1
                ))
                fig3.update_layout(
                    title='Feature Correlation Heatmap',
                    height=600
                )
                figures['correlation_heatmap'] = fig3.to_json()
            
            # 4. آمار تقلب بر اساس فروشنده
            if 'merchant' in current_data.columns and 'is_fraud' in current_data.columns:
                merchant_stats = current_data.groupby('merchant').agg({
                    'is_fraud': ['count', 'sum']
                }).reset_index()
                merchant_stats.columns = ['merchant', 'total', 'fraud']
                merchant_stats['fraud_rate'] = (merchant_stats['fraud'] / merchant_stats['total'] * 100).round(2)
                merchant_stats = merchant_stats.sort_values('fraud_rate', ascending=False).head(10)
                
                fig4 = go.Figure(data=[
                    go.Bar(
                        x=merchant_stats['merchant'],
                        y=merchant_stats['fraud_rate'],
                        text=merchant_stats['fraud_rate'],
                        textposition='auto',
                        marker_color='red'
                    )
                ])
                fig4.update_layout(
                    title='Top 10 Merchants by Fraud Rate',
                    xaxis_title='Merchant',
                    yaxis_title='Fraud Rate (%)'
                )
                figures['merchant_fraud_rate'] = fig4.to_json()
            
            return {
                'timestamp': datetime.utcnow().isoformat(),
                'figures': figures,
                'data_summary': {
                    'total_transactions': len(current_data),
                    'fraud_count': current_data['is_fraud'].sum() if 'is_fraud' in current_data.columns else 0,
                    'average_amount': current_data['amount'].mean(),
                    'max_amount': current_data['amount'].max(),
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Error creating visualizations: {e}")
            return {'error': str(e)}
    
    # ============== مانیتورینگ کامل ==============
    
    def run_complete_monitoring(self, days: int = 7) -> Dict[str, Any]:
        """
        اجرای مانیتورینگ کامل
        
        Args:
            days: تعداد روزهای گذشته
            
        Returns:
            دیکشنری شامل تمام گزارش‌ها
        """
        logger.info(f"🔄 Running complete monitoring for last {days} days")
        
        # بارگذاری داده‌های جاری
        current_data = self.load_current_data(days)
        
        if current_data.empty:
            return {
                'timestamp': datetime.utcnow().isoformat(),
                'status': 'error',
                'message': 'No data available for monitoring'
            }
        
        results = {
            'timestamp': datetime.utcnow().isoformat(),
            'period_days': days,
            'data_size': len(current_data),
            'reports': {}
        }
        
        # 1. Data Drift
        if self.reference_data is not None:
            drift_report = self.generate_data_drift_report(current_data)
            results['reports']['data_drift'] = drift_report
        
        # 2. Performance
        perf_report = self.generate_performance_report(current_data)
        results['reports']['performance'] = perf_report
        
        # 3. Target Drift
        if self.reference_data is not None:
            target_drift = self.generate_target_drift_report(current_data)
            results['reports']['target_drift'] = target_drift
        
        # 4. Data Quality
        quality_report = self.generate_data_quality_report(current_data)
        results['reports']['data_quality'] = quality_report
        
        # 5. Monitoring Tests
        tests = self.run_monitoring_tests(current_data)
        results['reports']['tests'] = tests
        
        # 6. Visualizations
        visualizations = self.create_visualization_dashboard(current_data)
        results['reports']['visualizations'] = visualizations
        
        # 7. خلاصه وضعیت
        status = 'healthy'
        alerts = []
        
        # بررسی drift
        if drift_report.get('drift_detected', False):
            status = 'warning'
            alerts.append({
                'type': 'data_drift',
                'severity': 'medium',
                'message': f"Data drift detected with score {drift_report.get('drift_score', 0)}"
            })
        
        # بررسی عملکرد
        f1_score = perf_report.get('f1_score', 0)
        if f1_score < 0.7:
            status = 'critical'
            alerts.append({
                'type': 'performance_degradation',
                'severity': 'high',
                'message': f"Model performance degraded: F1={f1_score:.4f}"
            })
        elif f1_score < 0.8:
            status = 'warning'
            alerts.append({
                'type': 'performance_degradation',
                'severity': 'low',
                'message': f"Model performance slightly degraded: F1={f1_score:.4f}"
            })
        
        results['status'] = status
        results['alerts'] = alerts
        
        # ذخیره نتایج
        results_path = self.monitoring_dir / f"monitoring_summary_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, default=str, indent=2)
        
        logger.info(f"✅ Monitoring complete: status={status}")
        
        return results

# ============== کلاس برای پایش بلادرنگ ==============

class RealtimeMonitor:
    """
    پایش بلادرنگ با استفاده از Kafka
    """
    
    def __init__(self, model_monitor: ModelMonitor):
        self.monitor = model_monitor
        self.buffer_size = 100
        self.data_buffer = []
        self.last_report_time = datetime.utcnow()
        self.report_interval_seconds = 300  # هر 5 دقیقه
        
    def process_transaction(self, transaction: Dict[str, Any]):
        """
        پردازش تراکنش جدید برای پایش
        """
        self.data_buffer.append(transaction)
        
        # اگر بافر پر شد، گزارش بگیر
        if len(self.data_buffer) >= self.buffer_size:
            self.generate_realtime_report()
        
        # بررسی زمان
        elapsed = (datetime.utcnow() - self.last_report_time).seconds
        if elapsed >= self.report_interval_seconds:
            self.generate_realtime_report()
    
    def generate_realtime_report(self):
        """
        تولید گزارش بلادرنگ
        """
        if not self.data_buffer:
            return
        
        try:
            # تبدیل به DataFrame
            current_data = pd.DataFrame(self.data_buffer)
            
            # تولید گزارش سریع
            report = {
                'timestamp': datetime.utcnow().isoformat(),
                'buffer_size': len(self.data_buffer),
                'fraud_count': current_data['is_fraud'].sum() if 'is_fraud' in current_data.columns else 0,
                'avg_amount': current_data['amount'].mean() if 'amount' in current_data.columns else 0,
                'reports': {}
            }
            
            # کیفیت داده
            quality = self.monitor.generate_data_quality_report(current_data)
            report['reports']['quality'] = quality
            
            # ارسال هشدار در صورت نیاز
            if quality.get('missing_values'):
                high_missing = {k: v for k, v in quality['missing_values'].items() if v > 0.1 * len(current_data)}
                if high_missing:
                    logger.warning(f"⚠️ High missing values: {high_missing}")
            
            # ذخیره گزارش
            report_path = self.monitoring_dir / f"realtime_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_path, 'w') as f:
                json.dump(report, f, default=str, indent=2)
            
            # پاک کردن بافر
            self.data_buffer = []
            self.last_report_time = datetime.utcnow()
            
            logger.info(f"📊 Realtime report generated: {len(current_data)} transactions")
            
        except Exception as e:
            logger.error(f"❌ Error generating realtime report: {e}")

# ============== مثال استفاده ==============

def main():
    """
    مثال استفاده از کلاس Monitoring
    """
    logger.info("=" * 60)
    logger.info("📊 Model Monitoring Demo")
    logger.info("=" * 60)
    
    # 1. ایجاد مانیتور
    monitor = ModelMonitor()
    
    # 2. اجرای مانیتورینگ کامل
    results = monitor.run_complete_monitoring(days=7)
    
    # 3. نمایش نتایج
    print("\n📊 Monitoring Results:")
    print(f"   Status: {results.get('status', 'unknown')}")
    print(f"   Data Size: {results.get('data_size', 0)}")
    print(f"   Alerts: {len(results.get('alerts', []))}")
    
    if results.get('alerts'):
        print("\n⚠️ Alerts:")
        for alert in results['alerts']:
            print(f"   - {alert['type']}: {alert['message']}")
    
    # 4. نمایش عملکرد مدل
    perf = results.get('reports', {}).get('performance', {})
    print("\n📈 Model Performance:")
    print(f"   Accuracy: {perf.get('accuracy', 'N/A')}")
    print(f"   F1 Score: {perf.get('f1_score', 'N/A')}")
    print(f"   ROC-AUC: {perf.get('roc_auc', 'N/A')}")
    
    print(f"\n✅ Reports saved in: {monitor.monitoring_dir}")

if __name__ == "__main__":
    main()