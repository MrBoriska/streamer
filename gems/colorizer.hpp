#pragma once

#include "engine/core/image/image.hpp"

namespace isaac {
void ImageF32ToHUEImageCuda(CudaImageConstView1f depth_image, CudaImageView3ub rgb_result,
                           float min_depth, float max_depth);

void ImageHUEToF32ImageCuda(CudaImageView3ub rgb_image, CudaImageView1f depth_result,
                           float min_depth, float max_depth);
}