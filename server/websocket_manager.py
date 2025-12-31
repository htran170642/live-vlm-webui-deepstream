"""
websocket_manager.py - Simple WebSocket connection management for VLM service
"""

import asyncio
import json
import logging
import os
import time
from typing import Dict, Optional
import uuid

import redis.asyncio as redis
from fastapi import WebSocket, WebSocketDisconnect

from models import VLMResult

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Simple WebSocket connection manager"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.connection_stats: Dict[str, dict] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str):
        """Accept new WebSocket connection"""
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.connection_stats[client_id] = {
            "connected_at": time.time(),
            "messages_sent": 0
        }
        logger.info(f"🔌 Client {client_id} connected. Total: {len(self.active_connections)}")
    
    def disconnect(self, client_id: str):
        """Remove WebSocket connection"""
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        if client_id in self.connection_stats:
            del self.connection_stats[client_id]
        logger.info(f"📱 Client {client_id} disconnected. Total: {len(self.active_connections)}")
    
    async def broadcast(self, message: str):
        """Broadcast message to all connected clients"""
        if not self.active_connections:
            return
        
        disconnected_clients = []
        
        for client_id, websocket in self.active_connections.items():
            try:
                await websocket.send_text(message)
                self.connection_stats[client_id]["messages_sent"] += 1
            except Exception as e:
                logger.warning(f"Failed to send to client {client_id}: {e}")
                disconnected_clients.append(client_id)
        
        # Clean up failed connections
        for client_id in disconnected_clients:
            self.disconnect(client_id)
        
        logger.debug(f"📤 Broadcasted to {len(self.active_connections)} clients")
    
    def get_connection_count(self) -> int:
        """Get number of active connections"""
        return len(self.active_connections)


class VLMWebSocketService:
    """Simple VLM streaming service"""
    
    def __init__(self, redis_host: str = "localhost", redis_port: int = 6379):
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.redis_client: Optional[redis.Redis] = None
        self.manager = ConnectionManager()
        self.is_running = False
        self.stream_name = "vlm:results:stream"  # Match C++ VLMRedisStreamManager
        self.last_id = "$"
        self.stats = {
            "total_messages_processed": 0,
            "service_start_time": time.time()
        }
        
        # Field mapping for DeepStream C++ Redis client (exact match)
        self.field_mapping = {
            "frame_number": ["frame_number", "frame", "frame_num"],  # DeepStream uses "frame_number"
            "source_id": ["source_id", "source", "camera_id"],      # DeepStream uses "source_id"
            "vlm_response": ["vlm_response", "vlm", "response"],     # DeepStream uses "vlm_response"
            "model_name": ["model_name", "model"],                  # DeepStream uses "model_name"
            "timestamp": ["timestamp", "time", "ts"]                # DeepStream uses "timestamp"
        }
    
    def get_field_value(self, fields: Dict[str, str], field_type: str, default_value=None):
        """Get field value using flexible field name mapping"""
        possible_names = self.field_mapping.get(field_type, [])
        
        for name in possible_names:
            if name in fields:
                return fields[name]
        
        return default_value
    
    async def connect_redis(self) -> bool:
        """Connect to Redis"""
        try:
            self.redis_client = redis.Redis(
                host=self.redis_host,
                port=self.redis_port,
                decode_responses=True
            )
            await self.redis_client.ping()
            logger.info(f"✅ Connected to Redis at {self.redis_host}:{self.redis_port}")
            return True
        except Exception as e:
            logger.error(f"❌ Redis connection failed: {e}")
            return False
    
    async def disconnect_redis(self):
        """Disconnect from Redis"""
        if self.redis_client:
            await self.redis_client.close()
            logger.info("🔴 Redis connection closed")
    
    async def start_streaming(self):
        """Start streaming VLM data from Redis"""
        if self.is_running:
            return
        
        self.is_running = True
        logger.info(f"🚀 Starting VLM stream monitoring: {self.stream_name}")
        
        while self.is_running:
            try:
                if not self.redis_client:
                    await self.connect_redis()
                    if not self.redis_client:
                        await asyncio.sleep(5)
                        continue
                
                # Read new messages from Redis stream
                messages = await self.redis_client.xread(
                    {self.stream_name: self.last_id},
                    block=1000,
                    count=10
                )
                
                if messages:
                    for stream, msgs in messages:
                        for msg_id, fields in msgs:
                            await self.process_vlm_message(msg_id, fields)
                            self.last_id = msg_id
                
            except redis.ConnectionError:
                logger.error("🔴 Redis connection lost, reconnecting...")
                self.redis_client = None
                await asyncio.sleep(5)
            except Exception as e:
                logger.error(f"❌ Stream processing error: {e}")
                await asyncio.sleep(1)
    
    async def process_vlm_message(self, msg_id: str, fields: Dict[str, str]):
        """Process VLM message and broadcast to clients"""
        try:
            # Parse VLM result using exact DeepStream field mapping
            vlm_result = VLMResult(
                message_id=msg_id,
                frame_number=int(self.get_field_value(fields, "frame_number", "0")),
                source_id=int(self.get_field_value(fields, "source_id", "0")),
                vlm_response=self.get_field_value(fields, "vlm_response", ""),
                model_name=self.get_field_value(fields, "model_name", "default"),
                timestamp=int(self.get_field_value(fields, "timestamp", str(int(time.time() * 1000)))),
                type=fields.get('type', 'vlm_result')
            )
            
            # Create WebSocket message
            ws_message = {
                "type": "vlm_result",
                "data": vlm_result.dict()
            }
            
            # Broadcast to all connected clients
            await self.manager.broadcast(json.dumps(ws_message))
            
            # Update stats
            self.stats["total_messages_processed"] += 1
            
            logger.debug(f"📤 Processed VLM: Frame {vlm_result.frame_number}, Source {vlm_result.source_id}")
            
        except Exception as e:
            logger.error(f"❌ Error processing VLM message {msg_id}: {e}")
            logger.error(f"❌ DeepStream fields: {fields}")  # Show fields only on error
    
    async def stop_streaming(self):
        """Stop streaming VLM data"""
        logger.info("🛑 Stopping VLM stream monitoring")
        self.is_running = False
    
    def get_service_stats(self) -> Dict:
        """Get basic service statistics"""
        current_time = time.time()
        return {
            "connected_clients": self.manager.get_connection_count(),
            "total_messages_processed": self.stats["total_messages_processed"],
            "uptime_seconds": int(current_time - self.stats["service_start_time"]),
            "redis_connected": self.redis_client is not None
        }
    
    async def handle_websocket_connection(self, websocket: WebSocket):
        """Handle WebSocket connection"""
        client_id = str(uuid.uuid4())
        
        try:
            await self.manager.connect(websocket, client_id)
            
            # Send welcome message
            welcome_msg = {
                "type": "connection",
                "message": "Connected to VLM stream",
                "client_id": client_id
            }
            await websocket.send_text(json.dumps(welcome_msg))
            
            # Keep connection alive
            while True:
                try:
                    data = await websocket.receive_text()
                    message = json.loads(data)
                    
                    # Handle ping/pong
                    if message.get("type") == "ping":
                        await websocket.send_text(json.dumps({"type": "pong"}))
                        
                except WebSocketDisconnect:
                    break
                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    logger.error(f"WebSocket error for client {client_id}: {e}")
                    break
                    
        except Exception as e:
            logger.error(f"WebSocket connection error: {e}")
        finally:
            self.manager.disconnect(client_id)