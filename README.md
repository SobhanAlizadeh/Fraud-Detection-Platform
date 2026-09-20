<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />

  <title>Fraud Detection Platform — Real-Time ML &amp; MLOps | Sobhan Alizadeh</title>
  <meta name="description" content="Production-ready end-to-end machine learning platform for real-time fraud detection — FastAPI, SHAP explainability, MLflow, Docker, Prometheus &amp; Grafana monitoring, PostgreSQL and CI/CD with GitHub Actions." />
  <meta name="keywords" content="Fraud Detection, Machine Learning, Real-Time Prediction, FastAPI, SHAP, MLflow, Prometheus, Grafana, Docker, Kubernetes, PostgreSQL, MLOps, Sobhan Alizadeh" />
  <meta name="author" content="Sobhan Alizadeh" />
  <meta name="robots" content="index, follow" />
  <meta name="theme-color" content="#2563EB" />
  <link rel="canonical" href="https://sobhanalizadeh.github.io/Fraud-Detection-Platform/" />

  <!-- Favicon -->
  <link rel="icon" type="image/svg+xml" href="favicon.svg" />
  <link rel="alternate icon" type="image/x-icon" href="favicon.ico" /> <!-- اختیاری: اگر favicon.ico ندارید حذف کنید -->

  <!-- Open Graph (پیش‌نمایش در شبکه‌های اجتماعی) -->
  <meta property="og:type" content="website" />
  <meta property="og:site_name" content="Sobhan Alizadeh" />
  <meta property="og:title" content="Fraud Detection Platform — Real-Time ML &amp; MLOps" />
  <meta property="og:description" content="End-to-end ML platform for real-time fraud detection with SHAP explainability, MLflow tracking and Prometheus &amp; Grafana monitoring." />
  <meta property="og:url" content="https://sobhanalizadeh.github.io/Fraud-Detection-Platform/" />
  <meta property="og:image" content="https://sobhanalizadeh.github.io/Fraud-Detection-Platform/og-image.png" /> <!-- اختیاری: تصویر 1200×630 بسازید؛ تا آن موقع این خط را حذف کنید -->

  <!-- Twitter Card -->
  <meta name="twitter:card" content="summary_large_image" />
  <meta name="twitter:title" content="Fraud Detection Platform — Real-Time ML &amp; MLOps" />
  <meta name="twitter:description" content="End-to-end ML platform for real-time fraud detection with SHAP explainability, MLflow tracking and Prometheus &amp; Grafana monitoring." />
  <meta name="twitter:image" content="https://sobhanalizadeh.github.io/Fraud-Detection-Platform/og-image.png" /> <!-- اختیاری -->

  <!-- JSON-LD — Structured Data for Google -->
  <script type="application/ld+json">
  {
    "@context": "https://schema.org",
    "@type": "WebApplication",
    "name": "Fraud Detection Platform",
    "url": "https://sobhanalizadeh.github.io/Fraud-Detection-Platform/",
    "description": "Production-ready end-to-end machine learning platform for real-time fraud detection — FastAPI, SHAP explainability, MLflow, Docker, Prometheus & Grafana monitoring, PostgreSQL and CI/CD with GitHub Actions.",
    "applicationCategory": "BusinessApplication",
    "operatingSystem": "Web",
    "inLanguage": "en",
    "author": {
      "@type": "Person",
      "name": "Sobhan Alizadeh",
      "jobTitle": "AI Engineer",
      "url": "https://github.com/sobhanalizadeh",
      "sameAs": [
        "https://github.com/sobhanalizadeh",
        "https://linkedin.com/in/sobhan-alizadeh"
      ]
    },
    "keywords": "Fraud Detection, Machine Learning, Real-Time Prediction, SHAP, MLflow, MLOps, FastAPI",
    "offers": { "@type": "Offer", "price": "0", "priceCurrency": "USD" }
  }
  </script>
</head>
# 🛡️ Fraud Detection Platform

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-green.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-24.0.7-blue.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![EvidentlyAI](https://img.shields.io/badge/EvidentlyAI-0.7.21-white.svg)](https://www.evidentlyai.com/)
[![Optuna](https://img.shields.io/badge/optuna-3.4.0-red.svg)](https://mlflow.org/)
[![MlFlow](https://img.shields.io/badge/MlFlow-2.8.1-brown.svg)](https://mlflow.org/)
[![Grafana](https://img.shields.io/badge/Grafana-10.2.0-orange.svg)](https://grafana.com/)

> **Real-time Fraud Detection System powered by Machine Learning**

A production-ready, end-to-end fraud detection platform built with FastAPI, Docker, and Machine Learning. Detect fraudulent transactions in real-time with monitoring, observability, and a beautiful dashboard.

---

# 📋 Table of Contents

- [✨ Features](#-features)
- [🏗️ Architecture](#️-architecture)
- [🚀 Quick Start](#-quick-start)
- [📊 Dashboard](#-dashboard)
- [📁 Project Structure](#-project-structure)
- [⚙️ Configuration](#️-configuration)
- [🔧 Development](#-development)
- [📈 Monitoring](#-monitoring)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)

---

# ✨ Features

## Core Features

- 🔍 **Real-time Fraud Detection** - Detect fraudulent transactions in milliseconds
- 🤖 **ML Models** - XGBoost, LightGBM, CatBoost
- 📊 **Data Pipeline** - ETL pipeline with data generation and validation
- 🗄️ **Feature Store** - Lightweight feature management

## Infrastructure

- 🐳 Dockerized deployment
- 📈 Prometheus & Grafana Monitoring
- 📊 Streamlit Dashboard
- 🧪 MLflow Experiment Tracking

## API & Integration

- 🚀 FastAPI
- 📝 Swagger/OpenAPI
- 🔄 Kafka Streaming
- 🗄️ PostgreSQL

---

# 🏗️ Architecture

```text
┌─────────────────────────────────────────────────────────────────────┐
│                     Client Applications                             │
├─────────────────────────────────────────────────────────────────────┤
│                  Streamlit Dashboard (8501)                         │
├─────────────────────────────────────────────────────────────────────┤
│                         FastAPI (8000)                              │
│ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐                  │
│ │   /detect    │ │   /history   │ │    /stats    │                  │
│ └──────────────┘ └──────────────┘ └──────────────┘                  │
│ ┌───────────────────────────────────────────────────────────────┐   │
│ │             FraudService (Business Logic)                     │   │
│ └───────────────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────────────┤
│                  Machine Learning Engine                           │
│ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐                  │
│ │ LightGBM     │ │   XGBoost    │ │  CatBoost    │                  │
│ └──────────────┘ └──────────────┘ └──────────────┘                  │
├─────────────────────────────────────────────────────────────────────┤
│                      Data & Storage                                │
│ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐                  │
│ │ PostgreSQL   │ │    Kafka     │ │    Models    │                  │
│ └──────────────┘ └──────────────┘ └──────────────┘                  │
├─────────────────────────────────────────────────────────────────────┤
│                     Monitoring Stack                                │
│ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ │
│ │  Prometheus  │ │   Grafana    │ │  Evidently   │ |    MlFlow    | │
│ └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

---

# 🚀 Quick Start

## Prerequisites

- Docker 24+
- Docker Compose 2.20+
- Python 3.11+

## Clone Repository

```bash
git clone https://github.com/yourusername/fraud-detection-platform.git
cd fraud-detection-platform
```

## Run Docker

```bash
docker-compose up -d

docker-compose ps

docker-compose logs -f api
```

## Available Services

| Service | URL |
|----------|-----|
| API | http://localhost:8000 |
| Swagger | http://localhost:8000/docs |
| Streamlit | http://localhost:8501 |
| Grafana | http://localhost:3000 |
| PGAdmin | http://localhost:5050 |
| Kafka UI | http://localhost:8080 |
| MlFlow | http://localhost:5000 |

---

## Test API

```bash
curl -X POST http://localhost:8000/api/v1/fraud/detect \
-H "Content-Type: application/json" \
-d '{
  "transaction_id":"TX123456",
  "user_id":"user_3851",
  "amount":1500.0,
  "timestamp":"2024-07-05T10:30:00",
  "merchant":"Amazon"
}'
```

```bash
curl http://localhost:8000/api/v1/fraud/stats?days=7
```

```bash
curl "http://localhost:8000/api/v1/fraud/history/all?days=7&limit=50"
```

---

# 📊 Dashboard

## Streamlit

- 📈 Overview
- 🔍 Real-time Detection
- 📊 Analytics
- 👤 User Risk Profile
- 📉 Model Performance

## Grafana

- API Metrics
- Model Performance
- Business Metrics

---

# 📁 Project Structure

```text
fraud-detection-platform/
├── src/
│   ├── api/
│   │   ├── routers/
│   │   ├── schemas/
│   │   └── services/
│   ├── core/
│   ├── data_engineering/
│   └── ml_engineering/
├── dashboard/
├── tests/
├── models/
├── data/
├── grafana/
├── monitoring_reports/
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

---

# ⚙️ Configuration

## Environment Variables

```bash
DB_HOST=postgres
DB_PORT=5432
DB_NAME=fraud_db
DB_USER=fraud_user
DB_PASSWORD=fraud_pass

API_HOST=0.0.0.0
API_PORT=8000
API_DEBUG=True

KAFKA_BOOTSTRAP_SERVERS=kafka:9092

MLFLOW_TRACKING_URI=http://mlflow:5000
```

---

# 🤖 Supported Models

| Model | Status |
|--------|--------|
| XGBoost | ✅ |
| LightGBM | ✅ |
| CatBoost | ✅ |

---

# 🔧 Development

## Local Setup

```bash
python3.11 -m venv venv

source venv/bin/activate

pip install -r requirements.txt

uvicorn src.api.main:app --reload

streamlit run dashboard/app.py
```

## Train Model

```bash
docker exec -it fraud_api python scripts/train_model.py
```

or

```bash
python scripts/train_model.py
```

## Run Tests

```bash
pytest tests/ -v --cov=src
```

## Generate Data

```bash
docker exec fraud_api python src/data_engineering/generator.py

docker exec fraud_api python src/data_engineering/etl_pipeline.py
```

---

# 📈 Monitoring

## Prometheus Metrics

- http_requests_total
- http_request_duration_seconds
- fraud_predictions_total
- model_accuracy

## Grafana Dashboards

- API Performance
- Fraud Analytics
- Model Performance
- System Health

---

# 🤝 Contributing

1. Fork repository
2. Create feature branch

```bash
git checkout -b feature/amazing-feature
```

3. Commit

```bash
git commit -m "Add amazing feature"
```

4. Push

```bash
git push origin feature/amazing-feature
```

5. Open Pull Request

---

# 📄 License

Licensed under the MIT License.

---

# 🙏 Acknowledgments

- FastAPI
- Scikit-learn
- Docker
- Grafana
- Streamlit

---
# 👨‍💻 Author

<p align="center">

<img src="https://github.com/SobhanAlizadeh.png" width="120" style="border-radius:50%"/>

</p>

<h2 align="center">Sobhan Alizadeh</h2>

<p align="center">
AI Engineer · Builder of Intelligent Systems · RAG & LLM Enthusiast
</p>

<p align="center">

<a href="https://github.com/SobhanAlizadeh">
<img src="https://img.shields.io/badge/GitHub-SobhanAlizadeh-181717?style=for-the-badge&logo=github"/>
</a>

<a href="https://linkedin.com/in/sobhan-alizadeh">
<img src="https://img.shields.io/badge/LinkedIn-Profile-0A66C2?style=for-the-badge&logo=linkedin"/>
</a>

</p>


---

# ⭐ Support

If you like this project, please give it a ⭐ on GitHub.

Built with ❤️ for Fraud Detection.
