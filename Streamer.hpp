#pragma once

#include <gst/gst.h>
#include <gst/app/gstappsrc.h>
#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <stdint.h>
#include <time.h>
#include <iostream>
#include <string>
#include <fstream>

#include <thread>

#include "engine/alice/alice_codelet.hpp"

#include "messages/camera.hpp"
#include "messages/math.hpp"

#include "messages/uuid.capnp.h"

#include <chrono>

namespace isaac {
namespace streaming {

// high frequency data (position)
typedef struct {

    // Timestamp
    std::chrono::system_clock::time_point timestamp;
    
    // Translation: [x,y,z]
    float trans[3];

    // Orientation [w,x,y,z]
    float quat[4];

} P3D;

class Streamer : public alice::Codelet {
  public:
    void start() override;
    void tick() override;
    void stop() override;

    ISAAC_PROTO_RX(ColorCameraProto, color);
    ISAAC_PROTO_RX(DepthCameraProto, depth);
    ISAAC_PROTO_RX(Pose3dProto, frame_position);
    ISAAC_PROTO_TX(DepthCameraProto, depth_debug);
    ISAAC_PROTO_TX(ColorCameraProto, depth_colorized_debug);

    ISAAC_PARAM(std::string, pipeline);
    ISAAC_PARAM(int, framerate, 30);

    void setCapsFromImage(GstAppSrc *appsrc, const ImageProto::Reader image_proto);
    // Creating a new klv buffer and send to the gstreamer pipeline
    void pushKLVBuffer(GstAppSrc *appsrc, Pose3dProto::Reader pose_proto, int64_t timestamp);
    // Creating a new buffer and to send to the gstreamer pipeline
    void pushBuffer(GstAppSrc *appsrc, const ImageConstView3ub rgb_image, int64_t timestamp);

    static gboolean gstError(GstBus *bus, GstMessage *message, gpointer userData);

    std::thread gst_thread;
    GMainLoop 	*loop		    = NULL;		// Main app loop keeps app alive
    GstElement	*pipeline	    = NULL;		// GStreamers pipeline for data flow
    GstElement	*appsrc_color   = NULL;		// Used to inject buffers into a pipeline
    GstElement	*appsrc_depth	= NULL;		// Used to inject buffers into a pipeline
    GstElement	*appsrc_data	= NULL;  // Used to inject buffers into a pipeline
    GError 		*error 		    = NULL;		// Holds error message if generated
};

class LatencyCalc : public alice::Codelet {
  public:
    void start() override;
    void tick() override;


    ISAAC_PROTO_RX(UuidProto, timestamp);
};
}  // namespace streaming
}  // namespace isaac

ISAAC_ALICE_REGISTER_CODELET(isaac::streaming::Streamer);
ISAAC_ALICE_REGISTER_CODELET(isaac::streaming::LatencyCalc);
