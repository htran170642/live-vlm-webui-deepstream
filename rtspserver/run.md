 ### Start RTSP server
 ```bash
 docker run --rm -it \
-e MTX_PROTOCOLS=tcp \
-p 8554:8554 \
bluenviron/mediamtx
```

### Start a RTSP from video
```bash
ffmpeg -re -stream_loop -1 -i /home/hieptt/Camera-31_record_2022-10-15_15-51-13.mp4 \
    -rtsp_transport tcp \
    -c copy \
    -f rtsp rtsp://localhost:8554/mystream
```