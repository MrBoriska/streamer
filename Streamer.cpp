#include "Streamer.hpp"

#include "engine/core/logger.hpp"
#include "engine/gems/image/utils.hpp"
#include "engine/gems/sight/sight.hpp"

#include "packages/streamer/gems/colorizer.hpp"
//#include "gstrealsensemeta.h"

namespace isaac {

namespace streaming {

void Streamer::start() {
    // Initialize GStreamer
    int argc = 0;
    char **argv;
    gst_init( &argc, &argv );

    // Create the main application loop.
    loop = g_main_loop_new( NULL, FALSE );

    // Builds the following pipeline.
    // Instruct GStreamer to construct the pipeline and get the beginning element appsrc.
    pipeline = gst_parse_launch( get_pipeline().c_str(), &error );
    if( !pipeline ) {
        reportFailure("GStreamer: %s", error->message);
        return;
    }
    
    gst_pipeline_set_latency(GST_PIPELINE(pipeline), guint(20));
    
    GstBus *bus = gst_pipeline_get_bus(GST_PIPELINE(pipeline));
    gst_bus_add_signal_watch(bus);
    g_signal_connect(bus, "message::error", G_CALLBACK(gstError), this);
    g_object_unref(bus);

    appsrc_color		= gst_bin_get_by_name( GST_BIN( pipeline ), "color" );
    appsrc_depth		= gst_bin_get_by_name( GST_BIN( pipeline ), "depth" );
    appsrc_data		    = gst_bin_get_by_name( GST_BIN( pipeline ), "data" );

    // Set Caps to data appsrc
    GstCaps *app_caps = gst_caps_new_simple("meta/x-klv", "parsed", G_TYPE_BOOLEAN, TRUE, NULL);
    gst_app_src_set_caps(GST_APP_SRC_CAST(appsrc_data), app_caps);
    gst_caps_unref(app_caps);

    // Set a few properties on the appsrc Element
    g_object_set( G_OBJECT( appsrc_color ), "is-live", TRUE, "format", GST_FORMAT_TIME, NULL );
    g_object_set( G_OBJECT( appsrc_depth ), "is-live", TRUE, "format", GST_FORMAT_TIME, NULL );
    g_object_set( G_OBJECT( appsrc_data ), "is-live", TRUE, "format", GST_FORMAT_TIME, NULL );

    // play
    gst_element_set_state( pipeline, GST_STATE_PLAYING );

    // Launch the stream in another thread
    gst_thread = std::thread([&]() {
        LOG_INFO("GMainLoop started.");
        // blocking
        g_main_loop_run(loop);
        // terminated!
        LOG_INFO("GMainLoop terminated.");
    });

    // Sync by timestamps
    synchronize(rx_color(), rx_depth());
    tickOnMessage(rx_color());
}

void Streamer::tick() {
    
    auto color_image_proto = rx_color().getProto();
    auto depth_image_proto = rx_depth().getProto();

    if (isFirstTick()) {
        // Create a caps (capabilities) struct that gets feed into the appsrc structure.
        setCapsFromImage(GST_APP_SRC( appsrc_color ), color_image_proto.getImage());
        setCapsFromImage(GST_APP_SRC( appsrc_depth ), depth_image_proto.getDepthImage());
    }

    if (!g_main_loop_is_running(loop)) return;

    // Copy Images from Proto into ImageConstView
    ImageConstView3ub color_image;
    bool ok = FromProto(color_image_proto.getImage(), rx_color().buffers(), color_image);
    ASSERT(ok, "Failed to deserialize the color image");
    CudaImageConstView1f cuda_depth_image;
    ok = FromProto(depth_image_proto.getDepthImage(), rx_depth().buffers(), cuda_depth_image);
    ASSERT(ok, "Failed to deserialize the depth image");

    CudaImage3ub cuda_depth_image_colorized(cuda_depth_image.rows(), cuda_depth_image.cols());
    ImageF32ToHUEImageCuda(cuda_depth_image, cuda_depth_image_colorized.view(), depth_image_proto.getMinDepth(), depth_image_proto.getMaxDepth());

    /*
    CudaImage1f cuda_depth_image_debug(cuda_depth_image_colorized.dimensions());
    ImageHUEToF32ImageCuda(cuda_depth_image_colorized.view(), cuda_depth_image_debug.view(), 0.4, 4.0);

    auto depth_image_debug_proto = tx_depth_debug().initProto();
    depth_image_debug_proto.setMinDepth(0.4);
    depth_image_debug_proto.setMaxDepth(4.0);
    ToProto(std::move(cuda_depth_image_debug), depth_image_debug_proto.initDepthImage(), tx_depth_debug().buffers());
    tx_depth_debug().publish();
    */

    // todo: need delete in future
    Image3ub depth_image_colorized(cuda_depth_image_colorized.dimensions());
    Copy(cuda_depth_image_colorized, depth_image_colorized);

    // Show Images in Sight
    show("framerate", 1/getTickDt());
    //show("image_color", [&](sight::Sop& sop) { sop.add(color_image); });
    //show("image_depth", [&](sight::Sop& sop) { sop.add(depth_image_colorized); });

    //show("latency_color", 0.000000001*(rx_color().acqtime() - rx_color().pubtime()));
    //show("latency_depth", 0.000000001*(rx_depth().acqtime() - rx_depth().pubtime()));
    
    // Push images into Gstreamer pipeline (appsrc)
    pushBuffer(GST_APP_SRC_CAST(appsrc_color), color_image, rx_color().pubtime());
    pushBuffer(GST_APP_SRC_CAST(appsrc_depth), depth_image_colorized, rx_depth().pubtime());
    
    if (rx_frame_position().available()) {
        auto pos_proto = rx_frame_position().getProto();
        pushKLVBuffer(GST_APP_SRC_CAST(appsrc_data), pos_proto, rx_frame_position().pubtime());
    }
}

void Streamer::setCapsFromImage(GstAppSrc *appsrc, const ImageProto::Reader image_proto) {
    GstCaps *app_caps = gst_caps_new_simple(
        "video/x-raw",
        "format", G_TYPE_STRING, "RGB",
        "width", G_TYPE_INT, image_proto.getCols(),
        "height", G_TYPE_INT, image_proto.getRows(),
        //"framerate", GST_TYPE_FRACTION, get_framerate(), 1,
        NULL
    );

    // This is going to specify the capabilities of the appsrc.
    gst_app_src_set_caps(appsrc, app_caps);

    // Don't need it anymore, un ref it so the memory can be removed.
    gst_caps_unref( app_caps );
}

void Streamer::pushKLVBuffer(GstAppSrc *appsrc, Pose3dProto::Reader pose_proto, uint64_t timestamp) {
    
    // Prepare data
    auto q = pose_proto.getRotation().getQ();
    auto t = pose_proto.getTranslation();

    P3D data;
    data.timestamp = std::chrono::time_point<std::chrono::high_resolution_clock>()+std::chrono::nanoseconds(timestamp);
    data.quat[0] = q.getW();
    data.quat[1] = q.getX();
    data.quat[2] = q.getY();
    data.quat[3] = q.getZ();
    data.trans[0] = t.getX();
    data.trans[1] = t.getY();
    data.trans[2] = t.getZ();
    
    // Create Buffer
    gsize size = sizeof(P3D);
    GstBuffer *buffer = gst_buffer_new();
    GstMemory *memory = gst_allocator_alloc(NULL, size, NULL);
    gst_buffer_insert_memory(buffer, -1, memory);
    gst_buffer_fill(buffer, 0, (gpointer)(&data), size);

    // Set Timestamp
    GST_BUFFER_TIMESTAMP(buffer) = timestamp;

    // Push buffer
    if (buffer == NULL) {
        reportFailure("gst_buffer_new_wrapped_full() returned NULL!");
    } else {
        // push buffer
        GstFlowReturn ret = gst_app_src_push_buffer(appsrc, buffer);
        if (ret < 0) {
            reportFailure("gst_app_src_push_buffer() returned error!");
        }
    }
}

void Streamer::pushBuffer(GstAppSrc *appsrc, const ImageConstView3ub rgb_image, uint64_t timestamp) {
    int size = rgb_image.num_elements();
    Image3ub to_gst_image(rgb_image.dimensions());
    Copy(rgb_image, to_gst_image);

    GstBuffer *buffer = gst_buffer_new();
    GstMemory *memory = gst_allocator_alloc(NULL, size, NULL);
    gst_buffer_insert_memory(buffer, -1, memory);
    gst_buffer_fill(buffer, 0, (gpointer)to_gst_image.data().pointer(), size);

    GST_BUFFER_TIMESTAMP(buffer) = timestamp;

    if (buffer == NULL) {
        reportFailure("gst_buffer_new_wrapped_full() returned NULL!");
    } else {
        GstFlowReturn ret = gst_app_src_push_buffer(appsrc, buffer);
        if (ret < 0) {
            reportFailure("gst_app_src_push_buffer() returned error!");
        }
    }
}

void Streamer::stop() {
    if (g_main_loop_is_running(loop)) {
        g_main_loop_quit(loop);
    }
    gst_element_set_state( pipeline, GST_STATE_NULL );
    gst_object_unref( GST_OBJECT ( pipeline ) );
    gst_object_unref( GST_OBJECT ( appsrc_color ) );
    gst_object_unref( GST_OBJECT ( appsrc_depth ) );
    g_main_loop_unref( loop );
    gst_thread.join();
}

gboolean Streamer::gstError(GstBus *bus, GstMessage *message, gpointer userData)
    {   
        Streamer *codelet = reinterpret_cast<Streamer*>(userData);
        
        GError *err;
        gchar *debug;
        gst_message_parse_error(message, &err, &debug);
        codelet->reportFailure("GStreamer: %s \n %s", err->message, debug);
        g_main_loop_quit(codelet->loop);
        g_error_free(err);
        g_free(debug);
        return FALSE;
    }
}  // namespace streaming
}  // namespace isaac
