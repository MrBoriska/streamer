#include <cuda_runtime.h>

#include "colorizer.cu.hpp"

namespace isaac {

namespace {

__global__ void ImageF32ToHUEImageImpl(StridePointer<const float> image,
                                       StridePointer<unsigned char> result,
                                       float min_depth, float max_depth,
                                       size_t width, size_t height) {
  
  unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= height || col >= width) return;
  
  char r,g,b;

  float d = image(row,col);
  if (min_depth <= d && d <= max_depth) {
    char dn = 1529*(d-min_depth)/(max_depth-min_depth);
    
    if (dn >= 0 && dn <= 255 || 1275 < dn && dn <= 1529)
      r = 255;
    else if (255 < dn && dn <= 510)
      r = 255 - dn;
    else if (510 < dn && dn <= 1020)
      r = 0;
    else if (1020 < dn && dn <= 1275)
      r = dn - 1020;
    
    if (0 < dn && dn <= 255)
      g = dn;
    else if (255 < dn && dn <= 510)
      g = 255;
    else if (510 < dn && dn <= 765)
      g = 765 - dn;
    else if (765 < dn && dn <= 1529)
      g = 0;
    
    if (0 < dn && dn <= 765)
      b = 0;
    else if (765 < dn && dn <= 1020)
      b = dn - 765;
    else if (1020 < dn && dn <= 1275)
      b = 255;
    else if (1275 < dn && dn <= 1529)
      b = 1275 - dn;
    
  } else {
    r = 0;
    g = 0;
    b = 0;
  }
  
  result(row, 3*col) = r;
  result(row, 3*col+1) = g;
  result(row, 3*col+2) = b;

  //unsigned char* data = result.row_pointer(row) + 3 * col;
  //data[0] = r;
  //data[1] = g;
  //data[2] = b;
}

}  // namespace

void ImageF32ToHUEImage(StridePointer<const float> image,
                        StridePointer<unsigned char> result,
                        float min_depth, float max_depth,
                        size_t width, size_t height) {
  // Split work into 16 by 16 grids across the images.
  dim3 block(16, 16);
  dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
  ImageF32ToHUEImageImpl<<<grid, block>>>(image, result, min_depth, max_depth, width, height);
}

}  // namespace isaac
