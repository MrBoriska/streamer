#include "cuda/colorizer.cu.hpp"
#include "colorizer.hpp"

namespace isaac {

void ImageF32ToHUEImageCuda(CudaImageConstView1f depth_image, CudaImageView3ub rgb_result, float min_depth, float max_depth) {

  ISAAC_ASSERT_EQ(depth_image.rows(), rgb_result.rows());
  ISAAC_ASSERT_EQ(depth_image.cols(), rgb_result.cols());
  ISAAC_ASSERT_EQ(1, depth_image.channels());
  ISAAC_ASSERT_EQ(3, rgb_result.channels());
  ImageF32ToHUEImage({depth_image.element_wise_begin(), depth_image.cols() * 1 * sizeof(float)},
                {rgb_result.element_wise_begin(), rgb_result.cols() * 3 * sizeof(uint8_t)},
                min_depth, max_depth, depth_image.cols(), depth_image.rows());
}
}