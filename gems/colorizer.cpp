#include "cuda/colorizer.cu.hpp"
#include "colorizer.hpp"

void ImageF32ToHUECuda(CudaImageConstView1f depth_image, CudaTensorView3ub rgb_result, float min_depth, float max_depth) {

  ISAAC_ASSERT_EQ(depth_image.rows(), rgb_result.dimensions()[0]);
  ISAAC_ASSERT_EQ(depth_image.cols(), rgb_result.dimensions()[1]);
  ISAAC_ASSERT_EQ(3, depth_image.dimensions()[2]);
  ImageF32ToHUEImage({depth_image.element_wise_begin(), depth_image.cols() * 1 * sizeof(float)},
                {rgb_result.element_wise_begin(), depth_image.cols() * 3 * sizeof(uint8_t)},
                min_depth, max_depth);
}