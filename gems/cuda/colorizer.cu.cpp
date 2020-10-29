#include <cuda_runtime.h>

#include "colorizer.cu.hpp"

namespace isaac {

namespace {

__global__ void ImageF32ToHUEImageImpl(StridePointer<const float> image,
                                       StridePointer<unsigned char> result,
                                       const float min_depth, const float max_depth,
                                       size_t width, size_t height) {
  
  unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= height || col >= width) return;
  
  unsigned char R = 0; // RED
  unsigned char G = 0; // GREEN
  unsigned char B = 0; // BLUE

  float d = image(row,col);
  if (min_depth <= d && d <= max_depth) {
    unsigned int dn = 1529.0f*(d-min_depth)/(max_depth-min_depth);
    
    if (dn <= 255 || 1275 < dn && dn <= 1529)  // 0 < dn <= 60 or 300 < dn < 360
      R = 255;
    else if (dn <= 510) // 60 < dn <= 120
      R = 510 - dn;
    else if (dn <= 1020) // 120 < dn <= 240
      R = 0;
    else if (dn <= 1275) // 240 < dn <= 300
      R = dn - 1020;
    
    if (dn <= 255) // 0 < dn <= 60
      G = dn;
    else if (dn <= 765) // 60 < dn <= 180
      G = 255;
    else if (dn <= 1020) // 180 < dn <= 240
      G = 765 - dn;
    else if (dn <= 1529) // 180 < dn <= 360
      G = 0;
    
    if (dn <= 510) // 0 < dn <= 120
      B = 0;
    else if (dn <= 765) // 120 < dn <= 180
      B = dn - 510;
    else if (dn <= 1275) // 180 < dn <= 300
      B = 255;
    else if (dn <= 1529) // 300 < dn <= 360
      B = 1529 - dn;
    
  }
  
  result(row, 3*col) = R;
  result(row, 3*col+1) = G;
  result(row, 3*col+2) = B;
}

__global__ void ImageHUEToF32ImageImpl(StridePointer<const unsigned char> image,
                                       StridePointer<float> result,
                                       const float min_depth, const float max_depth,
                                       size_t width, size_t height) {
  

  unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= height || col >= width) return;

  unsigned char R = image(row, 3*col);   // RED
  unsigned char G = image(row, 3*col+1); // GREEN
  unsigned char B = image(row, 3*col+2); // BLUE

  float dn = 0;

  if (B + G + R < 255) {
		dn = 0;
	} else if (R >= G && R >= B) { // {0, 60} or {300, 360}
		if (G >= B) {	// {0, 60}
			dn = G - B;
		} else { // {300, 360}
			dn = (G - B) + 1529;
		}
	} else if (G >= R && G >= B) { // {60, 180}
		dn = B - R + 510;
	} else if (B >= G && B >= R) { // {180, 300}
		dn = R - G + 1020;
	}
  
  result(row, col) = min_depth + (max_depth-min_depth)*dn/1529.0f;
}

}  // namespace

void ImageF32ToHUEImage(StridePointer<const float> image,
                        StridePointer<unsigned char> result,
                        const float min_depth, const float max_depth,
                        size_t width, size_t height) {
  // Split work into 16 by 16 grids across the images.
  dim3 block(16, 16);
  dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
  ImageF32ToHUEImageImpl<<<grid, block>>>(image, result, min_depth, max_depth, width, height);
}

void ImageHUEToF32Image(StridePointer<const unsigned char> image,
                        StridePointer<float> result,
                        float min_depth, float max_depth,
                        size_t width, size_t height) {
  dim3 block(16, 16);
  dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
  ImageHUEToF32ImageImpl<<<grid, block>>>(image, result, min_depth, max_depth, width, height);
  
}

}  // namespace isaac
