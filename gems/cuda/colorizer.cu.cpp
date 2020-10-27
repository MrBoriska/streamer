#include "colorizer.cu.hpp"
#include <cuda_runtime.h>

#include "engine/core/assert.hpp"

namespace isaac {

namespace {

__global__ void ImageF32ToHUEImageImpl(StridePointer<const float> image, StridePointer<unsigned char> result,
                                       float min_depth, float max_depth, unsigned int width, unsigned int height) {
  
  unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= height || col >= width) return;
 
  float dn = 1529.0*(image(row,col)-min_depth)/(max_depth-min_depth);
  
  char r,g,b;
  if (0 <= dn <= 255 || 1275 < dn <= 1529)
    r = 255;
  if (255 < dn <= 510)
    r = 255-dn;
  if (510<dn<=1020)
    r = 0;
  if (1020<dn<=1275)
    r = dn - 1020;
  
  if (0<dn<=255)
    g = dn;
  if (255<dn<=510)
    g = 255;
  if (510<dn<=765)
    g = 765-dn;
  if (765<dn<=1529)
    g = 0;
  
  if (0<dn<=765)
    b = 0;
  if (765<dn<=1020)
    b = dn-765;
  if (1020<dn<=1275)
    b = 255;
  if (1275<dn<=1529)
    b = 1275-dn;
  
  result(row, 3*col) = r;
  result(row, 3*col+1) = g;
  result(row, 3*col+2) = b;

  //unsigned char* data = result.row_pointer(row) + 3 * col;
  //data[0] = r;
  //data[1] = g;
  //data[2] = b;
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
