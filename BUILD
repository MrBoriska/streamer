"""
Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""

load("//engine/build:isaac.bzl", "isaac_app", "isaac_cc_module")

isaac_cc_module(
    name = "streamer",
    srcs = [
        "Streamer.cpp",# "gstrealsensemeta.cpp",
    ],
    hdrs = [
        "Streamer.hpp",# "gstrealsensemeta.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//packages/streamer/gems:colorizer",
        "@gstreamer",
        "@glib",
        "//engine/core/image",
        "//engine/core/math",
        "//engine/gems/sight",
    ],
)

isaac_cc_module(
    name = "reciever",
    srcs = [
        "Reciever.cpp"
    ],
    hdrs = [
        "Reciever.hpp"
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//packages/streamer/gems:colorizer",
        "@gstreamer",
        "@glib",
        "//engine/core/image",
        "//engine/core/math",
        "//engine/gems/sight",
    ],
)

isaac_app(
    name = "streamer",
    modules = [
        "//packages/streamer:streamer",
        "message_generators",
    ],
)

isaac_app(
    name = "reciever",
    modules = [
        "//packages/streamer:reciever",
        "message_generators",
    ],
)
