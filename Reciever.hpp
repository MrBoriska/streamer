#pragma once

#include <gst/gst.h>
#include <gst/app/gstappsrc.h>
#include <gst/app/gstappsink.h>
//#include "opencv2/core.hpp"
//#include "opencv2/highgui.hpp"
//#include "opencv2/imgproc.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <stdint.h>
#include <time.h>
#include <iostream>
#include <string>
#include <fstream>

#include <thread>

//#include <boost/thread.hpp>

#include "engine/alice/alice_codelet.hpp"
#include "engine/gems/image/color.hpp"
#include "messages/camera.hpp"
#include "messages/math.hpp"

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

class Reciever : public alice::Codelet {
  public:
    void start() override;
    void tick() override;
    void stop() override;

    ISAAC_PROTO_TX(ColorCameraProto, color);
    ISAAC_PROTO_TX(DepthCameraProto, depth);

    ISAAC_PARAM(std::string, pipeline);
    ISAAC_PARAM(int, framerate, 30);

    // Creating a new buffer and to send to the gstreamer pipeline
    static gboolean gstError(GstBus *bus, GstMessage *message, gpointer userData);
    static GstFlowReturn onNewColor(GstAppSink *appsink, gpointer userData);
    static GstFlowReturn onNewDepth(GstAppSink *appsink, gpointer userData);
    static GstFlowReturn onNewData(GstAppSink *appsink, gpointer userData);


    std::thread gst_thread;
    GMainLoop 	*loop		    = NULL;		// Main app loop keeps app alive
    GstElement	*pipeline	    = NULL;		// GStreamers pipeline for data flow
    GstElement	*appsink_color   = NULL;		// Used to inject buffers into a pipeline
    GstElement	*appsink_depth	= NULL;		// Used to inject buffers into a pipeline
    GstElement	*appsink_data	= NULL;		// Used to inject buffers into a pipeline
    GError 		*error 		    = NULL;		// Holds error message if generated
};
}  // namespace streaming
}  // namespace isaac

ISAAC_ALICE_REGISTER_CODELET(isaac::streaming::Reciever);
