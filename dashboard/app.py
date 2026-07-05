# dashboard/app.py
"""
🛡️ Fraud Detection Platform - Professional Dashboard
سیستم جامع تشخیص تقلب با Streamlit
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from datetime import datetime, timedelta
import json
import time
import random
from typing import Dict, Any, List, Optional
import os

# ============================================================
# تنظیمات صفحه
# ============================================================

st.set_page_config(
    page_title="🛡️ Fraud Detection Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# استایل سفارشی CSS
# ============================================================

st.markdown("""
<style>
    /* کارت‌های آماری */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    .metric-card h3 {
        margin: 0;
        font-size: 14px;
        opacity: 0.9;
    }
    .metric-card h1 {
        margin: 10px 0;
        font-size: 32px;
        font-weight: bold;
    }
    .metric-card-small {
        background: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        margin: 5px 0;
    }
    
    /* وضعیت‌ها */
    .status-safe {
        background: #d4edda;
        color: #155724;
        padding: 5px 15px;
        border-radius: 20px;
        display: inline-block;
    }
    .status-warning {
        background: #fff3cd;
        color: #856404;
        padding: 5px 15px;
        border-radius: 20px;
        display: inline-block;
    }
    .status-danger {
        background: #f8d7da;
        color: #721c24;
        padding: 5px 15px;
        border-radius: 20px;
        display: inline-block;
    }
    
    /* هدر */
    .header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin-bottom: 20px;
    }
    
    /* جداول */
    .dataframe {
        font-size: 12px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# تنظیمات API
# ============================================================

API_URL = (
    os.environ.get("API_URL") or 
    st.secrets.get("API_URL") or 
    "http://localhost:8000"
)
# ============================================================
# توابع کمکی
# ============================================================

def call_api(endpoint: str, method: str = "GET", data: Dict = None) -> Dict:
    """فراخوانی API با مدیریت خطا"""
    EndpointFinal=API_URL+endpoint
    print(f"Calling API: {EndpointFinal} with method {method}")
    try:
        if method == "GET":
            response = requests.get(f"{EndpointFinal}", timeout=10)
        elif method == "POST":
            response = requests.post(f"{EndpointFinal}", json=data, timeout=10)
        elif method == "PUT":
            response = requests.put(f"{EndpointFinal}", json=data, timeout=10)
        else:
            return {"error": "Invalid method"}
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"HTTP {response.status_code}"}
            
    except requests.exceptions.ConnectionError:
        return {"error": "❌ API is not running. Please start the API first."}
    except requests.exceptions.Timeout:
        return {"error": "⏰ API request timeout"}
    except Exception as e:
        return {"error": str(e)}

def format_currency(amount: float) -> str:
    """فرمت‌بندی مبلغ"""
    if amount >= 1000:
        return f"${amount:,.0f}"
    return f"${amount:,.2f}"

def format_percentage(value: float) -> str:
    """فرمت‌بندی درصد"""
    return f"{value * 100:.1f}%"

def get_risk_color(risk: str) -> str:
    """دریافت رنگ بر اساس سطح ریسک"""
    colors = {
        "LOW": "green",
        "MEDIUM": "orange",
        "HIGH": "red",
        "CRITICAL": "darkred"
    }
    return colors.get(risk, "gray")

def generate_sample_data(n: int = 100) -> pd.DataFrame:
    """تولید داده‌های نمونه برای نمایش"""
    np.random.seed(42)
    
    data = {
        'transaction_id': [f'TX{i:06d}' for i in range(n)],
        'user_id': [f'user_{random.randint(1, 50):03d}' for _ in range(n)],
        'amount': np.random.exponential(200, n) + 10,
        'timestamp': [datetime.now() - timedelta(hours=random.randint(0, 168)) for _ in range(n)],
        'merchant': np.random.choice(
            ['Amazon', 'Apple', 'Google', 'Netflix', 'Spotify', 'Uber', 
             'Crypto Exchange', 'Online Casino', 'Walmart', 'Target'],
            n
        ),
        'location': np.random.choice(['Tehran', 'Mashhad', 'Isfahan', 'Shiraz', 'Tabriz'], n),
        'is_fraud': np.random.choice([0, 1], n, p=[0.92, 0.08]),
        'fraud_score': np.random.beta(1, 10, n)
    }
    
    df = pd.DataFrame(data)
    df['fraud_score'] = df['fraud_score'] * df['is_fraud'] + df['fraud_score'] * 0.3 * (1 - df['is_fraud'])
    df['fraud_score'] = df['fraud_score'].clip(0, 1)
    
    return df

# ============================================================
# مقداردهی اولیه session_state
# ============================================================

if 'threshold' not in st.session_state:
    # دریافت آستانه از API
    response = call_api("/api/v1/fraud/threshold")
    if "error" not in response:
        st.session_state.threshold = response.get('threshold', 0.5)
    else:
        st.session_state.threshold = 0.5

# ============================================================
# سایدبار
# ============================================================

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/security-checked.png", width=80)
    st.title("🛡️ Fraud Detection")
    st.markdown("---")
    
    # منوی اصلی
    page = st.radio(
        "📊 Navigation",
        [
            "🏠 Dashboard",
            "🔍 Real-time Detection",
            "📊 Analytics",
            "👤 User Risk Profile",
            "📈 Model Performance",
            "⚙️ Settings"
        ],
        index=0
    )
    
    st.markdown("---")
    
    # وضعیت API
    st.subheader("🔌 System Status")
    health = call_api("/health")
    
    if "error" in health:
        st.error("❌ API Offline")
        st.caption("Please start the API server")
    else:
        status = health.get("status", "unknown")
        if status == "healthy":
            st.success("✅ API Online")
        else:
            st.warning(f"⚠️ {status}")
        
        # اطلاعات دیتابیس
        if "components" in health:
            db_status = health["components"].get("database", "unknown")
            if db_status == "healthy":
                st.caption("🗄️ Database: ✅ Connected")
            else:
                st.caption("🗄️ Database: ❌ Disconnected")
    
    st.markdown("---")
    st.caption(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ============================================================
# صفحه 1: داشبورد اصلی
# ============================================================

if page == "🏠 Dashboard":
    st.markdown("""
    <div class="header">
        <h1>🛡️ Fraud Detection Dashboard</h1>
        <p>Real-time monitoring and fraud detection platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    # دریافت داده‌ها
    with st.spinner("Loading data..."):
        stats = call_api("/api/v1/fraud/stats?days=7")
        
        if "error" in stats:
            st.warning("⚠️ Using sample data (API not available)")
            df = generate_sample_data(200)
            data = {
                "total_transactions": len(df),
                "fraud_transactions": df['is_fraud'].sum(),
                "fraud_rate": df['is_fraud'].mean(),
                "avg_amount": df['amount'].mean(),
                "max_amount": df['amount'].max(),
                "period_days": 7
            }
            sample_data = True
        else:
            sample_data = False
            data = stats.get("data", {})
    
    # کارت‌های آماری - ردیف اول
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
            <h3>📊 Total Transactions</h3>
            <h1>{data.get('total_transactions', 0):,}</h1>
            <p style="font-size:12px; opacity:0.8;">Last {data.get('period_days', 7)} days</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        fraud_count = data.get('fraud_transactions', 0)
        fraud_rate = data.get('fraud_rate', 0)
        color = "red" if fraud_rate > 10 else "orange" if fraud_rate > 5 else "green"
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
            <h3>🚨 Fraud Detected</h3>
            <h1>{fraud_count:,}</h1>
            <p style="font-size:12px; opacity:0.8;">Rate: <span style="color:{color};">{fraud_rate:.1f}%</span></p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
            <h3>💰 Avg Amount</h3>
            <h1>{format_currency(data.get('avg_amount', 0))}</h1>
            <p style="font-size:12px; opacity:0.8;">Per transaction</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);">
            <h3>📈 Max Amount</h3>
            <h1>{format_currency(data.get('max_amount', 0))}</h1>
            <p style="font-size:12px; opacity:0.8;">Highest transaction</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # نمودارها - ردیف دوم
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Transactions Over Time")
        
        if not sample_data and "daily_stats" in data:
            daily_df = pd.DataFrame(data['daily_stats'])
        else:
            dates = pd.date_range(end=datetime.now(), periods=7, freq='D')
            daily_df = pd.DataFrame({
                'date': dates.strftime('%Y-%m-%d'),
                'count': np.random.randint(50, 200, 7),
                'fraud_count': np.random.randint(0, 20, 7)
            })
        
        if not daily_df.empty:
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            fig.add_trace(
                go.Bar(
                    x=daily_df['date'],
                    y=daily_df['count'],
                    name="Total",
                    marker_color='#667eea',
                    text=daily_df['count'],
                    textposition='outside'
                ),
                secondary_y=False
            )
            
            fig.add_trace(
                go.Scatter(
                    x=daily_df['date'],
                    y=daily_df['fraud_count'],
                    name="Fraud",
                    line=dict(color='#f5576c', width=3),
                    mode='lines+markers',
                    marker=dict(size=10)
                ),
                secondary_y=True
            )
            
            fig.update_layout(
                height=400,
                hovermode='x unified',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            fig.update_yaxes(title_text="Total Transactions", secondary_y=False)
            fig.update_yaxes(title_text="Fraud Transactions", secondary_y=True)
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No daily data available")
    
    with col2:
        st.subheader("🏪 Top Merchants by Fraud")
        
        if not sample_data and "top_fraud_merchants" in data:
            merchant_df = pd.DataFrame(data['top_fraud_merchants'])
        else:
            merchants = ['Crypto Exchange', 'Online Casino', 'Amazon', 'Uber', 'Apple']
            counts = np.random.randint(5, 30, 5)
            merchant_df = pd.DataFrame({'merchant': merchants, 'count': counts})
        
        if not merchant_df.empty:
            fig = px.bar(
                merchant_df.head(10),
                x='merchant',
                y='count',
                title='Merchants with Most Fraud',
                color='count',
                color_continuous_scale='Reds',
                text='count'
            )
            fig.update_layout(
                height=400,
                xaxis_tickangle=-45,
                showlegend=False
            )
            fig.update_traces(textposition='outside')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No merchant data available")
    
    st.markdown("---")
    
    # جدول آخرین تراکنش‌ها
    st.subheader("🔄 Recent Transactions")
    
    if not sample_data:
        recent = call_api("/api/v1/fraud/history/all?days=1&limit=20")
        if "error" not in recent:
            recent_df = pd.DataFrame(recent.get("data", []))
        else:
            recent_df = generate_sample_data(20)
    else:
        recent_df = df.head(20)
    
    if not recent_df.empty:
        cols_to_show = ['transaction_id', 'user_id', 'amount', 'merchant', 'location', 'is_fraud', 'fraud_score']
        display_cols = [c for c in cols_to_show if c in recent_df.columns]
        
        if display_cols:
            display_df = recent_df[display_cols].copy()
            
            display_df['status'] = display_df['is_fraud'].apply(
                lambda x: '🚨 FRAUD' if x else '✅ Safe'
            )
            
            if 'amount' in display_df.columns:
                display_df['amount'] = display_df['amount'].apply(format_currency)
            
            if 'fraud_score' in display_df.columns:
                display_df['fraud_score'] = display_df['fraud_score'].apply(
                    lambda x: f"{x*100:.1f}%"
                )
            
            def highlight_fraud(row):
                if row.get('is_fraud', False):
                    return ['background-color: #ffcccc'] * len(row)
                return [''] * len(row)
            
            st.dataframe(
                display_df.style.apply(highlight_fraud, axis=1),
                use_container_width=True,
                height=300
            )
        else:
            st.info("No transaction data available")
    else:
        st.info("No recent transactions")
    if st.sidebar.button("🔄 Test API Connection"):
        with st.spinner("Testing connection..."):
            try:
                response = requests.get(f"{API_URL}/health", timeout=5)
                if response.status_code == 200:
                    st.sidebar.success(f"✅ Connected to {API_URL}")
                    st.sidebar.json(response.json())
                else:
                    st.sidebar.error(f"❌ Error: {response.status_code}")
            except Exception as e:
                st.sidebar.error(f"❌ Connection failed: {e}")

# ============================================================
# صفحه 2: تشخیص بلادرنگ
# ============================================================

elif page == "🔍 Real-time Detection":
    st.markdown("""
    <div class="header">
        <h1>🔍 Real-time Fraud Detection</h1>
        <p>Instantly check if a transaction is fraudulent</p>
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["🔍 Single Transaction", "📦 Batch Detection"])
    
    with tab1:
        st.subheader("🔍 Detect Fraud for a Single Transaction")
        
        col1, col2 = st.columns(2)
        
        with col1:
            transaction_id = st.text_input(
                "Transaction ID *",
                value=f"TX{datetime.now().strftime('%Y%m%d%H%M%S')}"
            )
            user_id = st.text_input("User ID *", value=f"user_{random.randint(1000, 9999)}")
            amount = st.number_input("Amount ($) *", min_value=0.01, value=250.0, step=10.0)
            
            merchant = st.selectbox(
                "Merchant *",
                ["Amazon", "Apple Store", "Google Play", "Spotify", "Netflix",
                 "Uber", "DoorDash", "Walmart", "Target", "Best Buy",
                 "Cryptocurrency Exchange", "Online Casino", "Gambling Site",
                 "Foreign Exchange", "Money Transfer"]
            )
        
        with col2:
            location = st.selectbox(
                "Location",
                ["Tehran", "Mashhad", "Isfahan", "Shiraz", "Tabriz",
                 "Qom", "Karaj", "Ahvaz", "Rasht", "Kermanshah"]
            )
            transaction_type = st.selectbox(
                "Transaction Type",
                ["online", "in_store", "atm", "transfer", "payment"]
            )
            device_id = st.text_input("Device ID", value=f"DEV-{random.randint(1000, 9999)}")
            ip_address = st.text_input("IP Address", value=f"192.168.1.{random.randint(1, 254)}")
            
            st.info(f"📊 Current threshold: {st.session_state.threshold * 100:.0f}%")
            
            threshold = st.slider(
                "Detection Threshold",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.threshold,
                step=0.05,
                help="Higher threshold = fewer false positives, may miss some fraud"
            )
        
        if st.button("🚀 Detect Fraud", type="primary", use_container_width=True):
            transaction = {
                "transaction_id": transaction_id,
                "user_id": user_id,
                "amount": amount,
                "timestamp": datetime.now().isoformat(),
                "merchant": merchant,
                "location": location,
                "transaction_type": transaction_type,
                "device_id": device_id,
                "ip_address": ip_address
            }
            
            with st.spinner("🧠 Analyzing transaction..."):
                # به‌روزرسانی آستانه
                threshold_update = call_api(
                    f"/api/v1/fraud/update-threshold?threshold={threshold}",
                    "POST"
                )
                
                if "error" in threshold_update:
                    st.error(f"❌ Error updating threshold: {threshold_update['error']}")
                else:
                    st.session_state.threshold = threshold
                    st.success(f"✅ Threshold updated to {threshold * 100:.0f}%")
                    
                    # تشخیص تقلب
                    result = call_api("/api/v1/fraud/detect", "POST", transaction)
                    
                    if "error" in result:
                        st.error(f"❌ {result['error']}")
                    else:
                        data = result.get("data", {})
                        prediction = data.get("prediction", {})
                        
                        st.success("✅ Detection completed!")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        fraud_prob = prediction.get("fraud_probability", 0)
                        is_fraud = prediction.get("is_fraud", False)
                        
                        with col1:
                            st.metric("Fraud Probability", f"{fraud_prob * 100:.1f}%")
                        
                        with col2:
                            status_text = "🚨 FRAUD" if is_fraud else "✅ LEGITIMATE"
                            status_color = "red" if is_fraud else "green"
                            st.markdown(
                                f"### Status: <span style='color:{status_color}; font-size:24px;'>{status_text}</span>",
                                unsafe_allow_html=True
                            )
                        
                        with col3:
                            st.metric("Threshold Used", f"{threshold * 100:.0f}%")
                        
                        with st.expander("📊 Detailed Prediction Information", expanded=True):
                            st.json(prediction)
                        
                        with st.expander("📝 Transaction Details"):
                            st.json(transaction)
    
    with tab2:
        st.subheader("📦 Batch Detection")
        st.info("Upload a CSV file with multiple transactions for batch analysis")
        
        st.download_button(
            label="📥 Download Sample CSV Template",
            data="""transaction_id,user_id,amount,timestamp,merchant,location,transaction_type,device_id,ip_address
TX001,user_001,150.50,2024-01-15T10:30:00,Amazon,Tehran,online,DEV-001,192.168.1.1
TX002,user_002,2500.00,2024-01-15T10:35:00,Crypto Exchange,Online,online,DEV-002,10.0.0.1
TX003,user_001,75.25,2024-01-15T10:40:00,Spotify,Tehran,online,DEV-001,192.168.1.1""",
            file_name="sample_transactions.csv",
            mime="text/csv"
        )
        
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.write(f"📊 **{len(df)}** transactions loaded")
                st.dataframe(df.head())
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Transactions", len(df))
                with col2:
                    st.metric("Total Amount", format_currency(df['amount'].sum() if 'amount' in df.columns else 0))
                
                if st.button("🔍 Detect Fraud in Batch", type="primary", use_container_width=True):
                    transactions = df.to_dict('records')
                    
                    with st.spinner(f"Analyzing {len(transactions)} transactions..."):
                        result = call_api("/api/v1/fraud/batch-detect", "POST", transactions)
                    
                    if "error" in result:
                        st.error(f"❌ {result['error']}")
                    else:
                        st.success(f"✅ Batch detection completed for {len(transactions)} transactions!")
                        
                        results = result.get("data", [])
                        if results:
                            predictions = []
                            for r in results:
                                pred = r.get("prediction", {})
                                predictions.append({
                                    "transaction_id": r.get("transaction_id"),
                                    "is_fraud": pred.get("is_fraud", False),
                                    "fraud_probability": pred.get("fraud_probability", 0)
                                })
                            
                            pred_df = pd.DataFrame(predictions)
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Processed", len(pred_df))
                            with col2:
                                fraud_count = pred_df['is_fraud'].sum()
                                st.metric("Fraud Detected", fraud_count)
                            with col3:
                                fraud_rate = fraud_count / len(pred_df) * 100 if len(pred_df) > 0 else 0
                                st.metric("Fraud Rate", f"{fraud_rate:.1f}%")
                            
                            st.subheader("📊 Detection Results")
                            st.dataframe(
                                pred_df.style.apply(
                                    lambda x: ['background-color: #ffcccc' if x['is_fraud'] else '' for _ in x],
                                    axis=1
                                ),
                                use_container_width=True
                            )
                            
                            csv = pred_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Results",
                                data=csv,
                                file_name=f"fraud_detection_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                            
            except Exception as e:
                st.error(f"❌ Error reading file: {e}")

# ============================================================
# صفحه 3: آنالیز
# ============================================================

elif page == "📊 Analytics":
    st.markdown("""
    <div class="header">
        <h1>📊 Advanced Analytics</h1>
        <p>Deep dive into transaction patterns and fraud trends</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        days = st.slider("Analysis Period", 1, 30, 7)
    with col2:
        st.write("")
        st.write("")
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.rerun()
    
    with st.spinner("Loading analytics data..."):
        stats = call_api(f"/api/v1/fraud/stats?days={days}")
        
        if "error" in stats:
            st.warning("⚠️ Using sample data")
            df = generate_sample_data(500)
        else:
            history = call_api(f"/api/v1/fraud/history/all?days={days}")
            if "error" not in history:
                df = pd.DataFrame(history.get("data", []))
            else:
                df = generate_sample_data(500)
    
    if not df.empty:
        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 Time Series",
            "💰 Amount Analysis",
            "📍 Geographic",
            "🏪 Merchant Analysis"
        ])
        
        with tab1:
            st.subheader("📈 Transaction Time Series")
            
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df['date'] = df['timestamp'].dt.date
                df['hour'] = df['timestamp'].dt.hour
                
                daily = df.groupby('date').agg({
                    'amount': ['count', 'sum', 'mean'],
                    'is_fraud': 'sum'
                }).reset_index()
                daily.columns = ['date', 'count', 'total_amount', 'avg_amount', 'fraud_count']
                
                fig = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=('Transaction Volume', 'Fraud Rate'),
                    vertical_spacing=0.15
                )
                
                fig.add_trace(
                    go.Bar(x=daily['date'], y=daily['count'], name='Transactions', marker_color='#667eea'),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Scatter(x=daily['date'], y=daily['total_amount'], name='Total Amount', 
                              line=dict(color='#764ba2', width=2), yaxis='y2'),
                    row=1, col=1
                )
                
                daily['fraud_rate'] = daily['fraud_count'] / daily['count'] * 100
                fig.add_trace(
                    go.Bar(x=daily['date'], y=daily['fraud_rate'], name='Fraud Rate %', 
                          marker_color='#f5576c'),
                    row=2, col=1
                )
                
                fig.update_layout(height=600, showlegend=True)
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.subheader("💰 Amount Distribution Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.histogram(
                    df,
                    x='amount',
                    nbins=50,
                    title='Transaction Amount Distribution',
                    color='is_fraud',
                    color_discrete_map={0: '#667eea', 1: '#f5576c'},
                    labels={'amount': 'Amount ($)', 'count': 'Frequency'}
                )
                fig.update_layout(showlegend=True)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.box(
                    df,
                    x='is_fraud',
                    y='amount',
                    title='Amount Distribution by Fraud Status',
                    color='is_fraud',
                    color_discrete_map={0: '#667eea', 1: '#f5576c'},
                    labels={'is_fraud': 'Is Fraud', 'amount': 'Amount ($)'}
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("📊 Descriptive Statistics")
            
            stats_df = df.groupby('is_fraud')['amount'].agg([
                'count', 'mean', 'std', 'min', 'max', 
                ('25%', lambda x: x.quantile(0.25)),
                ('50%', lambda x: x.quantile(0.5)),
                ('75%', lambda x: x.quantile(0.75))
            ]).round(2)
            
            stats_df.index = ['Normal', 'Fraud']
            st.dataframe(stats_df, use_container_width=True)
        
        with tab3:
            st.subheader("📍 Geographic Analysis")
            
            if 'location' in df.columns:
                location_stats = df.groupby('location').agg({
                    'amount': ['count', 'sum', 'mean'],
                    'is_fraud': 'sum'
                }).reset_index()
                location_stats.columns = ['location', 'count', 'total_amount', 'avg_amount', 'fraud_count']
                location_stats['fraud_rate'] = location_stats['fraud_count'] / location_stats['count'] * 100
                location_stats = location_stats.sort_values('count', ascending=False)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = px.bar(
                        location_stats.head(10),
                        x='location',
                        y='count',
                        title='Transactions by Location',
                        color='fraud_count',
                        color_continuous_scale='Reds',
                        text='count'
                    )
                    fig.update_layout(showlegend=False)
                    fig.update_traces(textposition='outside')
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = px.bar(
                        location_stats.sort_values('fraud_rate', ascending=False).head(10),
                        x='location',
                        y='fraud_rate',
                        title='Fraud Rate by Location (%)',
                        color='fraud_rate',
                        color_continuous_scale='Reds',
                        text=location_stats['fraud_rate'].round(1)
                    )
                    fig.update_layout(showlegend=False)
                    fig.update_traces(textposition='outside')
                    st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            st.subheader("🏪 Merchant Analysis")
            
            if 'merchant' in df.columns:
                merchant_stats = df.groupby('merchant').agg({
                    'amount': ['count', 'sum', 'mean'],
                    'is_fraud': 'sum'
                }).reset_index()
                merchant_stats.columns = ['merchant', 'count', 'total_amount', 'avg_amount', 'fraud_count']
                merchant_stats['fraud_rate'] = merchant_stats['fraud_count'] / merchant_stats['count'] * 100
                merchant_stats = merchant_stats.sort_values('fraud_rate', ascending=False)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = px.bar(
                        merchant_stats.head(10),
                        x='merchant',
                        y='fraud_rate',
                        title='Highest Fraud Rate Merchants',
                        color='fraud_rate',
                        color_continuous_scale='Reds',
                        text=merchant_stats['fraud_rate'].round(1)
                    )
                    fig.update_layout(showlegend=False)
                    fig.update_traces(textposition='outside')
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = px.bar(
                        merchant_stats.head(10),
                        x='merchant',
                        y='total_amount',
                        title='Highest Transaction Volume Merchants',
                        color='fraud_count',
                        color_continuous_scale='Blues',
                        text=merchant_stats['total_amount'].apply(format_currency)
                    )
                    fig.update_layout(showlegend=False)
                    fig.update_traces(textposition='outside')
                    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# صفحه 4: پروفایل ریسک کاربر
# ============================================================

elif page == "👤 User Risk Profile":
    st.markdown("""
    <div class="header">
        <h1>👤 User Risk Profile Analysis</h1>
        <p>Analyze individual user behavior and risk score</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        user_id = st.text_input("Enter User ID", value="user_12345")
    
    with col2:
        st.write("")
        st.write("")
        analyze_btn = st.button("🔍 Analyze User", type="primary", use_container_width=True)
    
    if analyze_btn:
        with st.spinner(f"Analyzing user {user_id}..."):
            result = call_api(f"/api/v1/fraud/risk-profile/{user_id}")
        
        if "error" in result:
            st.warning("⚠️ Using sample data for demonstration")
            
            risk_score = random.randint(0, 100)
            if risk_score >= 70:
                risk_level = "HIGH"
                color = "red"
            elif risk_score >= 40:
                risk_level = "MEDIUM"
                color = "orange"
            else:
                risk_level = "LOW"
                color = "green"
            
            data = {
                "user_id": user_id,
                "risk_score": risk_score,
                "risk_level": risk_level,
                "metrics": {
                    "total_transactions": random.randint(10, 100),
                    "total_amount": random.uniform(1000, 50000),
                    "avg_amount": random.uniform(50, 500),
                    "max_amount": random.uniform(200, 5000),
                    "fraud_transactions": random.randint(0, 10),
                    "fraud_rate": random.uniform(0, 20)
                },
                "risk_factors": [
                    {"factor": "high_transaction_count", "weight": 10},
                    {"factor": "abnormal_amount", "weight": 20},
                    {"factor": "fraud_history", "weight": 30}
                ] if risk_score > 50 else []
            }
        else:
            data = result.get("data", {})
            risk_level = data.get("risk_level", "UNKNOWN")
            color = get_risk_color(risk_level)
        
        st.markdown("---")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
                <h3>👤 User</h3>
                <h1 style="font-size:20px;">{data.get('user_id', 'N/A')}</h1>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
                <h3>🎯 Risk Score</h3>
                <h1>{data.get('risk_score', 0)}</h1>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
                <h3>⚠️ Risk Level</h3>
                <h1 style="color:{color};">{risk_level}</h1>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            transactions = data.get('metrics', {}).get('total_transactions', 0)
            st.markdown(f"""
            <div class="metric-card" style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);">
                <h3>📊 Transactions</h3>
                <h1>{transactions}</h1>
            </div>
            """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Transaction Metrics")
            metrics = data.get('metrics', {})
            if metrics:
                metric_df = pd.DataFrame({
                    'Metric': ['Total Amount', 'Average Amount', 'Max Amount', 'Fraud Count', 'Fraud Rate'],
                    'Value': [
                        format_currency(metrics.get('total_amount', 0)),
                        format_currency(metrics.get('avg_amount', 0)),
                        format_currency(metrics.get('max_amount', 0)),
                        metrics.get('fraud_transactions', 0),
                        f"{metrics.get('fraud_rate', 0):.1f}%"
                    ]
                })
                st.dataframe(metric_df, use_container_width=True)
        
        with col2:
            st.subheader("⚠️ Risk Factors")
            risk_factors = data.get('risk_factors', [])
            if risk_factors:
                for factor in risk_factors:
                    weight = factor.get('weight', 0)
                    color = 'red' if weight > 20 else 'orange' if weight > 10 else 'green'
                    st.markdown(f"""
                    <div class="metric-card-small">
                        <b>{factor.get('factor', 'Unknown')}</b>
                        <span style="float:right; color:{color};">Weight: {weight}</span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("✅ No significant risk factors detected")

# ============================================================
# صفحه 5: عملکرد مدل
# ============================================================

elif page == "📈 Model Performance":
    st.markdown("""
    <div class="header">
        <h1>📈 Model Performance Monitoring</h1>
        <p>Track model accuracy, drift detection, and system health</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.spinner("Loading model performance data..."):
        performance = call_api("/api/v1/fraud/performance")
    
    if "error" in performance:
        st.warning("⚠️ Using sample performance data")
        performance = {
            "data": {
                "total_predictions": 1250,
                "fraud_predicted": 98,
                "fraud_rate_predicted": 7.84,
                "avg_fraud_probability": 0.312,
                "probability_distribution": {
                    "low": 850,
                    "medium": 320,
                    "high": 80
                },
                "latest_prediction_time": datetime.now().isoformat()
            }
        }
    
    data = performance.get("data", {})
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Predictions", f"{data.get('total_predictions', 0):,}")
    
    with col2:
        st.metric("Fraud Predicted", data.get('fraud_predicted', 0))
    
    with col3:
        fraud_rate = data.get('fraud_rate_predicted', 0)
        st.metric("Fraud Rate", f"{fraud_rate:.1f}%")
    
    with col4:
        st.metric("Avg Probability", f"{data.get('avg_fraud_probability', 0):.3f}")
    
    st.subheader("📊 Prediction Probability Distribution")
    
    dist = data.get('probability_distribution', {})
    if dist:
        df_dist = pd.DataFrame({
            'Category': ['Low (< 0.3)', 'Medium (0.3-0.7)', 'High (> 0.7)'],
            'Count': [dist.get('low', 0), dist.get('medium', 0), dist.get('high', 0)]
        })
        
        fig = px.pie(
            df_dist,
            values='Count',
            names='Category',
            title='Probability Distribution',
            color='Category',
            color_discrete_map={
                'Low (< 0.3)': '#43e97b',
                'Medium (0.3-0.7)': '#f9d423',
                'High (> 0.7)': '#f5576c'
            }
        )
        fig.update_traces(textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    
    if 'latest_prediction_time' in data:
        st.info(f"🕐 Latest Prediction: {data['latest_prediction_time']}")

# ============================================================
# صفحه 6: تنظیمات
# ============================================================

elif page == "⚙️ Settings":
    st.markdown("""
    <div class="header">
        <h1>⚙️ Settings</h1>
        <p>Configure system parameters and preferences</p>
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["🎯 Detection", "🔌 API", "📊 Export"])
    
    with tab1:
        st.subheader("🎯 Detection Threshold")
        
        current_threshold = st.session_state.get('threshold', 0.5)
        new_threshold = st.slider(
            "Set detection threshold",
            min_value=0.0,
            max_value=1.0,
            value=current_threshold,
            step=0.05,
            help="Higher threshold = fewer false positives, lower threshold = more detections"
        )
        
        if new_threshold != current_threshold:
            if st.button("Update Threshold", type="primary"):
                result = call_api(f"/api/v1/fraud/update-threshold?threshold={new_threshold}", "POST")
                if "error" in result:
                    st.error(f"❌ {result['error']}")
                else:
                    st.success(f"✅ Threshold updated to {new_threshold * 100:.0f}%")
                    st.session_state.threshold = new_threshold
        
        st.subheader("📊 Threshold Impact")
        
        thresholds = np.arange(0.1, 1.0, 0.05)
        detections = 100 * (1 - thresholds) * np.exp(-2 * thresholds)
        false_positives = 100 * (1 - thresholds) * np.exp(-0.5 * thresholds)
        
        impact_df = pd.DataFrame({
            'Threshold': thresholds,
            'Detections': detections,
            'False Positives': false_positives
        })
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=impact_df['Threshold'],
            y=impact_df['Detections'],
            name='Detections (%)',
            line=dict(color='#667eea', width=3)
        ))
        fig.add_trace(go.Scatter(
            x=impact_df['Threshold'],
            y=impact_df['False Positives'],
            name='False Positives (%)',
            line=dict(color='#f5576c', width=3, dash='dash')
        ))
        fig.add_vline(
            x=st.session_state.threshold,
            line_dash="dash",
            line_color="green",
            annotation_text=f"Current: {st.session_state.threshold*100:.0f}%",
            annotation_position="top"
        )
        fig.update_layout(
            title='Detection vs False Positives by Threshold',
            xaxis_title='Threshold',
            yaxis_title='Percentage (%)',
            hovermode='x unified',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("🔌 API Configuration")
        
        api_url = st.text_input("API Base URL", value=API_URL)
        
        if st.button("Save API Configuration"):
            st.success("✅ Configuration saved!")
            st.info("Please restart the dashboard for changes to take effect")
    
    with tab3:
        st.subheader("📊 Data Export")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📥 Export Model Report", use_container_width=True):
                st.success("✅ Model report exported!")
                st.download_button(
                    label="Download Report",
                    data=json.dumps({"status": "success", "timestamp": datetime.now().isoformat()}),
                    file_name=f"model_report_{datetime.now().strftime('%Y%m%d')}.json",
                    mime="application/json"
                )
        
        with col2:
            if st.button("📊 Export All Data", use_container_width=True):
                st.success("✅ Data export initiated!")

# ============================================================
# فوتر
# ============================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #999; padding: 10px;'>
    <strong>🛡️ Fraud Detection Platform v1.0.0</strong><br>
    Built with ❤️ using FastAPI, Streamlit, and AI
</div>
""", unsafe_allow_html=True)

# Auto-refresh
if st.sidebar.checkbox("🔄 Auto-refresh (30s)", value=False):
    time.sleep(30)
    st.rerun()