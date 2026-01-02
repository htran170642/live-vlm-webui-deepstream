# Live VLM WebUI with Deepstream

## Project Components

1. DeepStream OOP Pipeline
2. RTSP server
3. Websocket service
4. VLM Analytics Dashboard  
5. VLM service

## Quick Setup

### 1. Build DeepStream Pipeline
```bash
cd deepstream
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### 2. Start Video Analytics
```bash
# Start RTSP streams
docker run --rm -it -e MTX_PROTOCOLS=tcp -p 8554:8554 bluenviron/mediamtx
ffmpeg -re -stream_loop -1 -i video.mp4 -rtsp_transport tcp -c copy -f rtsp rtsp://localhost:8554/stream


# Start DeepStream with VLM analytics
cd deepstream
./build/deepstream-test-app dstest3_config.yaml
```
### 3. Start Redis and Websocket service
```bash
docker-compose up
```

### 4. Monitor with Dashboard
```bash
# Open VLM dashboard
cd ui
open index.html
# Click "Connect VLM" to start monitoring
```

## Architecture

```
RTSP Streams → DeepStream Pipeline → VLM Processing → WebSocket → Dashboard
     ↓              ↓                    ↓             ↓           ↓
Video Files → Object Detection → AI Analysis → Real-time → Browser UI
```

### Future works
#### Your real-time flow NEVER stops
```
Redis Stream → Parse VLM → Enqueue (0.5ms) → WebSocket Broadcast
                                 ↓
                           Background Worker → Database (batched)

```

## Requirements

- **DeepStream SDK 7.0+** - NVIDIA video analytics
- **CUDA Toolkit** - GPU acceleration  
- **GStreamer 1.0** - Media framework
- **CMake 3.10+** - Build system
- **FastAPI service** - VLM WebSocket on `localhost:8000`


Perfect for security monitoring, industrial surveillance, smart city applications, or any multi-camera AI analytics system.
