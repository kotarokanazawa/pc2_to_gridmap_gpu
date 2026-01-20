// gpu_gridmap.cu
// GPU kernel: per-cell MAX height (zmax) using ordered-int + atomicMax

#include <cuda_runtime.h>
#include <stdint.h>
#include <math_constants.h>

__device__ __forceinline__ int floatToOrderedInt(float f)
{
  int i = __float_as_int(f);
  return (i >= 0) ? i : (i ^ 0x7fffffff);
}

__global__ void pointsToMaxHeightKernelOrderedInt(
    const float* __restrict__ xs,
    const float* __restrict__ ys,
    const float* __restrict__ zs,
    int n,
    float origin_x, float origin_y,
    float resolution,
    int width, int height,
    int* __restrict__ grid_max_zi)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  float x = xs[i];
  float y = ys[i];
  float z = zs[i];

  if (!isfinite(x) || !isfinite(y) || !isfinite(z)) return;

  int ix = (int)floorf((x - origin_x) / resolution);
  int iy = (int)floorf((y - origin_y) / resolution);

  if ((unsigned)ix >= (unsigned)width || (unsigned)iy >= (unsigned)height) return;

  int idx = iy * width + ix;
  int zi  = floatToOrderedInt(z);

  atomicMax(&grid_max_zi[idx], zi);
}

extern "C" void launch_points_to_grid_maxheight_orderedint(
    const float* d_xs,
    const float* d_ys,
    const float* d_zs,
    int n,
    float origin_x, float origin_y,
    float resolution,
    int width, int height,
    int* d_grid_max_zi,
    cudaStream_t stream)
{
  int threads = 256;
  int blocks  = (n + threads - 1) / threads;
  pointsToMaxHeightKernelOrderedInt<<<blocks, threads, 0, stream>>>(
      d_xs, d_ys, d_zs, n, origin_x, origin_y, resolution, width, height, d_grid_max_zi);
}
