#include <cuda_runtime.h>

#include "engine/gems/cuda_utils/stride_pointer.hpp"

namespace isaac {

// Converts a 1-channel 32-bit floating point depth image to a 3-channel 8-bit HUE image.
void ImageF32ToHUEImage(StridePointer<const float> image,
                        StridePointer<unsigned char> result,
                        float min_depth, float max_depth,
                        size_t width, size_t height);

}  // namespace isaac
