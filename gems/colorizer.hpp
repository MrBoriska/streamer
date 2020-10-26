
#include "engine/core/math/utils.hpp"
#include "engine/gems/image/color.hpp"

void ImageF32ToHUEImageCuda(CudaImageConstView1f depth_image, CudaTensorView3ub rgb_result,
                           float min_depth, float max_depth);