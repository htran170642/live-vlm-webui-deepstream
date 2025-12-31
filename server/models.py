"""
models.py - Simple data models for VLM WebSocket service
"""

from typing import Optional
from pydantic import BaseModel


class VLMResult(BaseModel):
    """VLM analysis result from DeepStream"""
    message_id: str
    frame_number: int
    source_id: int
    vlm_response: str
    model_name: str = "default"
    timestamp: int
    type: str = "vlm_result"


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    service: str
    redis_status: str
    connected_clients: int
    total_messages_processed: int
    uptime_seconds: int
    timestamp: int


class ServiceStats(BaseModel):
    """Basic service statistics"""
    connected_clients: int
    total_messages_processed: int
    uptime_seconds: int
    redis_connected: bool