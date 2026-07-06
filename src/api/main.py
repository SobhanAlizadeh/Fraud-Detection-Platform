# src/api/main.py
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse  # <--- اضافه شده
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from contextlib import asynccontextmanager
import logging
import uvicorn
from datetime import datetime
import time

from api.services import consumer_service

from .routers import fraud
from .schemas.transaction import ErrorResponse
from ..core.config import config
from ..core.database import db_manager

from prometheus_client import Counter, Histogram, generate_latest, REGISTRY
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
# ============== تنظیمات Logging ==============

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)



app = FastAPI(title="Fraud Detection API")


# متریک‌ها
REQUESTS = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration', ['method', 'endpoint'])

# Middleware
@app.middleware('http')
async def prometheus_middleware(request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    
    REQUESTS.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    
    DURATION.labels(
        method=request.method,
        endpoint=request.url.path
    ).observe(duration)
    
    return response

# Endpoint
@app.get('/metrics')
async def metrics():
    return Response(generate_latest(REGISTRY), media_type='text/plain')
# ============== مدیریت چرخه حیات ==============

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Starting Fraud Detection API...")
    try:
        db_manager.create_tables()
        logger.info("✅ Database tables created/verified")
    except Exception as e:
        logger.error(f"❌ Startup error: {e}")
    yield
    logger.info("👋 Shutting down...")
    db_manager.close()
    try:
        consumer_service.start()
        logger.info("✅ Kafka consumer started")
    except Exception as e:
        logger.error(f"❌ Failed to start Kafka consumer: {e}")
    
    yield
    
    # SHUTDOWN
    logger.info("👋 Shutting down...")
    consumer_service.stop()

# ============== ایجاد اپلیکیشن ==============

app = FastAPI(
    title="Fraud Detection Platform API",
    description="🛡️ سیستم تشخیص تقلب هوشمند",
    version="1.0.0",
    lifespan=lifespan
)

# ============== Middleware ==============

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    logger.info(f"📥 {request.method} {request.url.path}")
    
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        logger.info(f"📤 {request.method} {request.url.path} - Status: {response.status_code} - Time: {process_time:.3f}s")
        return response
    except Exception as e:
        logger.error(f"❌ Error processing {request.url.path}: {e}")
        raise

# ============== روترها ==============

app.include_router(fraud.router)

# ============== هندلرهای خطا ==============

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.error(f"HTTP Exception: {exc.status_code} - {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "error": f"HTTP {exc.status_code}",
            "detail": str(exc.detail),
            "timestamp": datetime.utcnow().isoformat()
        }
    )

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled Exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "error": "Internal Server Error",
            "detail": str(exc) if config.api.debug else "An unexpected error occurred",
            "timestamp": datetime.utcnow().isoformat()
        }
    )

# ============== اندپوینت‌های عمومی ==============

@app.get("/", tags=["General"])
async def root():
    return {
        "name": "Fraud Detection Platform API",
        "version": "1.0.0",
        "status": "operational",
        "timestamp": datetime.utcnow().isoformat(),
        "documentation": "/docs",
        "health": "/health"
    }

@app.get("/health", tags=["General"])
async def health_check():
    """
    بررسی سلامت سرویس
    """
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "components": {
            "api": "healthy",
            "database": "unknown",
            "models": "unknown"
        }
    }
    
    # بررسی دیتابیس با متد جدید
    try:
        if db_manager.test_connection():
            health_status["components"]["database"] = "healthy"
        else:
            health_status["components"]["database"] = "unhealthy: Connection failed"
            health_status["status"] = "degraded"
    except Exception as e:
        health_status["components"]["database"] = f"unhealthy: {str(e)}"
        health_status["status"] = "degraded"
    
    # بررسی مدل‌ها (اختیاری)
    try:
        from ..ml_engineering.inference import FraudInferenceService
        inference = FraudInferenceService()
        if inference.model is not None:
            health_status["components"]["models"] = "healthy"
        else:
            health_status["components"]["models"] = "degraded (model not loaded)"
            health_status["status"] = "degraded"
    except Exception as e:
        health_status["components"]["models"] = f"unhealthy: {str(e)}"
        health_status["status"] = "degraded"
    
    return health_status
@app.get("/info", tags=["General"])
async def get_info():
    return {
        "application": {
            "name": "Fraud Detection Platform",
            "version": "1.0.0",
            "environment": "development" if config.api.debug else "production"
        },
        "api": {
            "docs": "/docs",
            "redoc": "/redoc"
        },
        "database": {
            "host": config.db.host,
            "database": config.db.database
        },
        "timestamp": datetime.utcnow().isoformat()
    }

# ============== اجرا ==============

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=config.api.host,
        port=config.api.port,
        reload=config.api.debug
    )