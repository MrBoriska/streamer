"""
Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""

load("//engine/build:cc_cuda_library.bzl", "cc_cuda_library")
load("//engine/build:isaac.bzl", "isaac_app", "isaac_cc_module", "isaac_cc_library")

isaac_cc_module(
    name = "streamer",
    srcs = [
        "Streamer.cpp",
    ],
    hdrs = [
        "Streamer.hpp",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//packages/streamer:colorizer",
        "@gstreamer",
        "@glib",
        "//engine/core/image",
        "//engine/core/math",
        "//engine/core/tensor",
        "//engine/gems/sight",
    ],
)

isaac_cc_module(
    name = "reciever",
    srcs = [
        "Reciever.cpp",
    ],
    hdrs = [
        "Reciever.hpp",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "@gstreamer",
        "@glib",
        "//engine/core/image",
        "//engine/core/math",
        "//engine/core/tensor",
        "//engine/gems/sight",
    ],
)

isaac_cc_library(
    name = "colorizer",
    srcs = [
        "gems/colorizer.cpp",
    ],
    hdrs = [
        "gems/colorizer.hpp",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//packages/streamer:cuda_colorizer",
        "//engine/core",
        "//engine/core/math",
        "//engine/core/image",
        "//engine/gems/image",
    ],
)

cc_cuda_library(
    name = "cuda_colorizer",
    srcs = [
        "gems/cuda/colorizer.cu.cpp",
    ],
    hdrs = [
        "gems/cuda/colorizer.cu.hpp",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//engine/core",
        "//engine/gems/cuda_utils",
        "//third_party:cudart",
    ],
)

isaac_app(
    name = "streamer",
    modules = [
        "//packages/streamer:streamer",
        "message_generators",
    ],
)
