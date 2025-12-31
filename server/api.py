"""
api.py - Simple FastAPI application for VLM WebSocket service
"""

import asyncio
import logging
import os
import time

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

from models import HealthResponse, ServiceStats
from websocket_manager import VLMWebSocketService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize VLM service with environment variables
vlm_service = VLMWebSocketService(
    redis_host=os.environ.get("REDIS_HOST", "localhost"),
    redis_port=int(os.environ.get("REDIS_PORT", "6379"))
)

# Initialize FastAPI application
app = FastAPI(
    title="VLM WebSocket Service",
    description="Simple WebSocket streaming of VLM results from DeepStream",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Application lifecycle events
@app.on_event("startup")
async def startup_event():
    """Initialize service on startup"""
    logger.info("🚀 Starting VLM WebSocket Service")
    logger.info(f"🔗 Redis configuration: {vlm_service.redis_host}:{vlm_service.redis_port}")
    
    # Connect to Redis
    if await vlm_service.connect_redis():
        # Start streaming task
        vlm_service.stream_task = asyncio.create_task(vlm_service.start_streaming())
        logger.info("✅ VLM service initialized successfully")
    else:
        logger.error("❌ Failed to initialize VLM service")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("🛑 Shutting down VLM WebSocket Service")
    await vlm_service.stop_streaming()
    await vlm_service.disconnect_redis()


# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for VLM data streaming"""
    await vlm_service.handle_websocket_connection(websocket)


# REST API endpoints
@app.get("/")
async def get_root():
    """Root endpoint with service information"""
    return {
        "service": "VLM WebSocket Service",
        "version": "1.0.0", 
        "description": "Simple WebSocket streaming of VLM results from DeepStream",
        "websocket_endpoint": "/ws",
        "health_check": "/health",
        "statistics": "/stats"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    service_stats = vlm_service.get_service_stats()
    
    return HealthResponse(
        status="healthy",
        service="VLM WebSocket Service",
        redis_status="connected" if service_stats["redis_connected"] else "disconnected",
        connected_clients=service_stats["connected_clients"],
        uptime_seconds=service_stats["uptime_seconds"],
        total_messages_processed=service_stats["total_messages_processed"],
        timestamp=int(time.time() * 1000)
    )


@app.get("/stats", response_model=ServiceStats)
async def get_stats():
    """Get basic service statistics"""
    service_stats = vlm_service.get_service_stats()
    
    return ServiceStats(
        connected_clients=service_stats["connected_clients"],
        total_messages_processed=service_stats["total_messages_processed"],
        uptime_seconds=service_stats["uptime_seconds"],
        redis_connected=service_stats["redis_connected"]
    )


# Entry point for running the application
if __name__ == "__main__":
    import uvicorn
    
    uvicorn_config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
    
    server = uvicorn.Server(uvicorn_config)
    
    redis_host = os.environ.get("REDIS_HOST", "localhost")
    redis_port = os.environ.get("REDIS_PORT", "6379")
    
    print("🚀 Starting Simple VLM WebSocket Service")
    print(f"🔗 Redis: {redis_host}:{redis_port}")
    print("📊 API Documentation: http://localhost:8000/docs")
    print("🔌 WebSocket Endpoint: ws://localhost:8000/ws")
    print("❤️  Health Check: http://localhost:8000/health")
    print("📊 Statistics: http://localhost:8000/stats")
    print("")
    print("💡 Environment variables:")
    print(f"   REDIS_HOST={redis_host}")
    print(f"   REDIS_PORT={redis_port}")
    
    try:
        server.run()
    except KeyboardInterrupt:
        logger.info("👋 Service stopped by user")