#include "cuda/colorizer.cu.hpp"
#include "colorizer.hpp"

namespace isaac {

void ImageF32ToHUEImageCuda(CudaImageConstView1f depth_image, CudaImageView3ub rgb_result, float min_depth, float max_depth) {

  ISAAC_ASSERT_EQ(depth_image.rows(), rgb_result.rows());
  ISAAC_ASSERT_EQ(depth_image.cols(), rgb_result.cols());
  ISAAC_ASSERT_EQ(1, depth_image.channels());
  ISAAC_ASSERT_EQ(3, rgb_result.channels());

  ImageF32ToHUEImage({depth_image.element_wise_begin(), depth_image.getStride()},
                {rgb_result.element_wise_begin(), rgb_result.getStride()},
                min_depth, max_depth, depth_image.cols(), depth_image.rows());
}

void ImageHUEToF32ImageCuda(CudaImageView3ub rgb_image, CudaImageView1f depth_result, float min_depth, float max_depth) {

  ISAAC_ASSERT_EQ(rgb_image.rows(), depth_result.rows());
  ISAAC_ASSERT_EQ(rgb_image.cols(), depth_result.cols());
  ISAAC_ASSERT_EQ(1, depth_result.channels());
  ISAAC_ASSERT_EQ(3, rgb_image.channels());

  ImageHUEToF32Image({rgb_image.element_wise_begin(), rgb_image.getStride()},
                {depth_result.element_wise_begin(), depth_result.getStride()},
                min_depth, max_depth, rgb_image.cols(), rgb_image.rows());
}
}