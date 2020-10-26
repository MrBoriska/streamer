#include "colorizer.cu.hpp"
#include <cuda_runtime.h>

#include "engine/core/assert.hpp"

namespace isaac {

namespace {

__global__ void ImageF32ToHUEImageImpl(StridePointer<const float> image, StridePointer<unsigned char> result,
                                       float min_depth, float max_depth, unsigned int width, unsigned int height) {
  


}

}  // namespace

void ImageF32ToHUEImage(StridePointer<const float> image, StridePointer<unsigned char> result,
                        float min_depth, float max_depth, unsigned int width, unsigned int height) {
  // Split work into 16 by 16 grids across the images.
  dim3 block(16, 16);
  dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
  ImageF32ToHUEImageImpl<<<grid, block>>>(image, result, min_depth, max_depth, width, height);
}

}  // namespace isaac
