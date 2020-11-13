# RGBD data stream over GStreamer

## Send (config.pipeline):

```
appsrc name=depth ! videoconvert ! video/x-raw,format=I420 !
    nvvidconv output-buffers=10 ! video/x-raw(memory:NVMM),format=NV12 ! m.sink_0
appsrc name=color ! videoconvert ! video/x-raw,format=I420 !
    nvvidconv output-buffers=10 ! video/x-raw(memory:NVMM),format=NV12 ! m.sink_1
nvcompositor name=m start-time-selection=1
    sink_0::xpos=0 sink_0::ypos=0
    sink_1::xpos=0 sink_1::ypos={1} \
    sink_0::width={0} sink_0::height={1}
    sink_1::width={2} sink_1::height={3} !
        nvvidconv output-buffers=90 ! video/x-raw(memory:NVMM),format=NV12 !
        omxh264enc preset-level=1 control-rate=2 bitrate=8000000 ! video/x-h264,stream-format=byte-stream !
        mux. 
appsrc name=data ! mux. 
mpegtsmux name=mux ! udpsink host=192.168.9.100 port=5000  buffer-size=5344160 sync=false
```

Connect RGBD to `Streamer/depth` and `Streamer/color` RX Hooks.
The sizes for appsrc caps are set automatically, but the framerate must be set in `config.framerate`.


## Recieve:

```
udpsrc port=5000 caps = \"video/mpegts, systemstream=(boolean)true, packetsize=(int)188\" buffer-size=5344160 \
    ! tsdemux name=dm dm. ! queue ! h264parse ! avdec_h264 ! tee name=t \
    t. ! queue ! videocrop top=0 right=432 left=0 bottom=720 ! videoconvert ! video/x-raw,format=(string)RGB ! appsink name=depth \
    t. ! queue ! videocrop top=480 right=0 left=0 bottom=0 ! videoconvert ! video/x-raw,format=(string)RGB ! appsink name=color \
    dm. ! queue ! meta/x-klv ! appsink name=data
```