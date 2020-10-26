# RGBD data stream over GStreamer

## Send (config.pipeline):

```
appsrc name=depth ! videoconvert ! video/x-raw,format=I420 ! \
    nvvidconv output-buffers=10 ! video/x-raw(memory:NVMM),format=NV12 ! m.sink_0 \
appsrc name=color ! videoconvert ! video/x-raw,format=I420 ! \
    nvvidconv output-buffers=10 ! video/x-raw(memory:NVMM),format=NV12 ! m.sink_1 \
nvcompositor name=m start-time-selection=1 \
    sink_0::xpos=0 sink_0::ypos=0 \
    sink_1::xpos=0 sink_1::ypos=720 \
    sink_0::width=1280 sink_0::height=720 \
    sink_1::width=1280 sink_1::height=720 !\
nvvidconv ! video/x-raw(memory:NVMM),format=NV12 !\
omxh264enc ! video/x-h264,stream-format=byte-stream ! rtph264pay pt=96 ! udpsink host=123.123.123.123 port=5000
```

Connect RGBD to `Streamer/depth` and `Streamer/color` RX Hooks.
The sizes for appsrc caps are set automatically, but the framerate must be set in `config.framerate`.


## Recieve:

```
gst-launch-1.0 -v  udpsrc port=5000 caps = "application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)H264, payload=(int)96" ! rtph264depay ! avdec_h264 ! videoconvert ! "video/x-raw,format=(string)I420" ! autovideosink
```