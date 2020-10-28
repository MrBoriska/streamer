
load("//engine/build:cc_cuda_library.bzl", "cc_cuda_library")
load("//engine/build:isaac.bzl", "isaac_cc_module", "isaac_cc_library")

isaac_cc_library(
    name = "colorizer",
    srcs = [
        "colorizer.cpp",
    ],
    hdrs = [
        "colorizer.hpp",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//engine/core",
        "//engine/core/image",
        "//packages/streamer/gems:cuda_colorizer",
    ],
)

cc_cuda_library(
    name = "cuda_colorizer",
    srcs = [
        "cuda/colorizer.cu.cpp",
    ],
    hdrs = [
        "cuda/colorizer.cu.hpp",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//engine/core",
        "//engine/gems/cuda_utils",
        "//third_party:cudart",
    ],
)