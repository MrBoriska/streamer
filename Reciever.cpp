#include "Reciever.hpp"

#include "engine/core/logger.hpp"
#include "engine/gems/image/conversions.hpp"
#include "engine/gems/image/utils.hpp"
#include "engine/gems/sight/sight.hpp"

#include "gstrealsensemeta.h"

namespace isaac {

namespace streaming {

void Reciever::start() {
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

    GstBus *bus = gst_pipeline_get_bus(GST_PIPELINE(pipeline));
    gst_bus_add_signal_watch(bus);
    g_signal_connect(bus, "message::error", G_CALLBACK(gstError), this);
    g_object_unref(bus);

    appsink_color		= gst_bin_get_by_name( GST_BIN( pipeline ), "color" );
    appsink_depth		= gst_bin_get_by_name( GST_BIN( pipeline ), "depth" );

    // Set a few properties on the appsrc Element
    g_object_set(G_OBJECT(appsink_depth), "emit-signals", true, "max-buffers", 1, NULL);
    g_object_set(G_OBJECT(appsink_color), "emit-signals", true, "max-buffers", 1, NULL);
    g_signal_connect(G_OBJECT(appsink_depth), "new-sample", G_CALLBACK(onNewDepth), this);
    g_signal_connect(G_OBJECT(appsink_color), "new-sample", G_CALLBACK(onNewColor), this);

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
    tickPeriodically();
}

void Reciever::tick() {
    if (!g_main_loop_is_running(loop)) return;

    // Show Images in Sight
    //show("image_color", [&](sight::Sop& sop) { sop.add(color_image); });
    //show("image_depth", [&](sight::Sop& sop) { sop.add(depth_image); });
    
    // Push images into Gstreamer pipeline (appsrc)
}

GstFlowReturn Reciever::onNewColor (GstAppSink *appsink, gpointer userData) {
    Reciever *codelet = reinterpret_cast<Reciever*>(userData);

    GstSample *sample;
    g_signal_emit_by_name(appsink, "pull-sample", &sample);


    // Get image size from caps
    GstCaps *caps = gst_sample_get_caps(sample);
    uint caps_size = gst_caps_get_size(caps);
    gint width, height;
    for (uint i = 0; i < caps_size; ++i) {
        GstStructure *s = gst_caps_get_structure(caps, i);
        gst_structure_get_int(s, "width", &width);
        gst_structure_get_int(s, "height", &height);
    };
    
    // Get buffer
    GstMapInfo map;
    GstBuffer *buffer = gst_sample_get_buffer(sample);
    gst_buffer_map(buffer, &map, GST_MAP_READ);

    // Convert to Isaac SDK ImageProto
    CpuBufferConstView image_buffer(reinterpret_cast<const byte*>(map.data), map.size);
    ImageConstView3ub color_image_view(image_buffer, height, width);
    Image3ub color_image(color_image_view.dimensions());
    Copy(color_image_view, color_image);

    auto color_image_proto = codelet->tx_color().initProto();
    color_image_proto.setColorSpace(ColorCameraProto::ColorSpace::RGB);
    ToProto(std::move(color_image), color_image_proto.initImage(), codelet->tx_color().buffers());

    //memcpy( &msg.data[0], map.data, map.size );

    codelet->tx_color().publish();

    gst_buffer_unmap(buffer, &map);
    gst_sample_unref(sample);

    return GST_FLOW_OK;
}

GstFlowReturn Reciever::onNewDepth (GstAppSink *appsink, gpointer userData) {
    Reciever *codelet = reinterpret_cast<Reciever*>(userData);
    GstSample *sample;
    g_signal_emit_by_name(appsink, "pull-sample", &sample);


    // Get image size from caps
    GstCaps *caps = gst_sample_get_caps(sample);
    uint caps_size = gst_caps_get_size(caps);
    gint width, height;
    for (uint i = 0; i < caps_size; ++i) {
        GstStructure *s = gst_caps_get_structure(caps, i);
        gst_structure_get_int(s, "width", &width);
        gst_structure_get_int(s, "height", &height);
    };
    
    // Get buffer
    GstMapInfo map;
    GstBuffer *buffer = gst_sample_get_buffer(sample);

    gst_buffer_map(buffer, &map, GST_MAP_READ);

    // Try get Metadata
    GstRealsenseMeta *meta = gst_buffer_get_realsense_meta(buffer);
    if (meta != nullptr)
        LOG_WARNING("Model: %s\n Serial: %s\n DepthUnits: %f", meta->cam_model, meta->cam_serial_number, meta->depth_units);
    else
        LOG_ERROR("Metadata is NULL");
    

    // Convert to Isaac SDK ImageProto
    CpuBufferConstView image_buffer(reinterpret_cast<const byte*>(map.data), map.size);
    ImageConstView3ub color_image_view(image_buffer, height, width);
    Image3ub color_image(color_image_view.dimensions());
    Copy(color_image_view, color_image);

    auto color_image_proto = codelet->tx_depth().initProto();
    color_image_proto.setColorSpace(ColorCameraProto::ColorSpace::RGB);
    ToProto(std::move(color_image), color_image_proto.initImage(), codelet->tx_depth().buffers());

    //memcpy( &msg.data[0], map.data, map.size );

    codelet->tx_depth().publish();

    gst_buffer_unmap(buffer, &map);
    gst_sample_unref(sample);

    return GST_FLOW_OK;
}

void Reciever::stop() {
    if (g_main_loop_is_running(loop)) {
        g_main_loop_quit(loop);
    }
    gst_element_set_state( pipeline, GST_STATE_NULL );
    gst_object_unref( GST_OBJECT ( pipeline ) );
    gst_object_unref( GST_OBJECT ( appsink_color ) );
    gst_object_unref( GST_OBJECT ( appsink_depth ) );
    g_main_loop_unref( loop );
    gst_thread.join();
}

gboolean Reciever::gstError(GstBus *bus, GstMessage *message, gpointer userData)
    {   
        Reciever *codelet = reinterpret_cast<Reciever*>(userData);
        
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
