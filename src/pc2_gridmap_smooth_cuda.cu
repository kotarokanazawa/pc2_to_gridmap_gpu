// pc2_gridmap_smooth_cuda.cu

#include <cuda_runtime.h>
#include <stdint.h>
#include <math_constants.h>

#include <vector>
#include <limits>
#include <cmath>
#include <cstring>
#include <algorithm>

#include "pc2_to_gridmap_gpu/pc2_gridmap_smooth_cuda.hpp"

// 外部（gpu_gridmap.cu）に既にある launcher を使う
extern "C" void launch_points_to_grid_maxheight_orderedint(
    const float* d_xs,
    const float* d_ys,
    const float* d_zs,
    int n,
    float origin_x, float origin_y,
    float resolution,
    int width, int height,
    int* d_grid_max_zi,
    cudaStream_t stream);

namespace pc2_gridmap_smooth
{

static inline bool cudaOk(cudaError_t e) { return e == cudaSuccess; }

// host-side ordered-int helper (MUST match gpu_gridmap.cu)
static inline int32_t floatToOrderedIntHost(float f)
{
  int32_t i;
  std::memcpy(&i, &f, sizeof(float));
  return (i >= 0) ? i : (i ^ 0x7fffffff);
}

// ---------------- device helpers ----------------
__device__ __forceinline__ bool pc2_gms_isfinite(float v)
{
  return isfinite(v);
}

// ordered-int -> float (device only)
__device__ __forceinline__ float orderedIntToFloatDevice(int32_t oi)
{
  int32_t raw = (oi >= 0) ? oi : (oi ^ 0x7fffffff);
  return __int_as_float(raw);
}

// ---------------- kernels ----------------

__global__ void pc2_gms_init_int(int32_t* a, int N, int32_t v)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  a[i] = v;
}

// mask out of z-range points by NaN (so existing kernel will skip via isfinite)
__global__ void pc2_gms_mask_points_by_z(
    float* xs, float* ys, float* zs, int n, float zmin, float zmax)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  float z = zs[i];
  if (!pc2_gms_isfinite(z) || z < zmin || z > zmax) {
    xs[i] = NAN;
    ys[i] = NAN;
    zs[i] = NAN;
  }
}

// ordered-int grid -> float grid (empty -> NaN)
__global__ void pc2_gms_orderedint_to_float(
    const int32_t* grid_oi, float* grid_z, int N, int32_t empty_oi)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;

  int32_t oi = grid_oi[i];
  if (oi == empty_oi) {
    grid_z[i] = NAN;
    return;
  }

  grid_z[i] = orderedIntToFloatDevice(oi);
}

// naive box smoothing (NaN-ignore mean)
__global__ void pc2_gms_smooth_box_nanmean(
    const float* in, float* out,
    int width, int height, int R)
{
  int iy = blockIdx.x * blockDim.x + threadIdx.x;
  if (iy >= height) return;

  for (int ix = 0; ix < width; ++ix)
  {
    float s = 0.0f;
    int n = 0;

    int x0 = max(0, ix - R);
    int x1 = min(width  - 1, ix + R);
    int y0 = max(0, iy - R);
    int y1 = min(height - 1, iy + R);

    for (int y = y0; y <= y1; ++y) {
      int base = y * width;
      for (int x = x0; x <= x1; ++x) {
        float v = in[base + x];
        if (pc2_gms_isfinite(v)) { s += v; n++; }
      }
    }

    int idx = iy * width + ix;
    out[idx] = (n > 0) ? (s / (float)n) : in[idx];
  }
}

// ---------------- CudaPipeline ----------------

CudaPipeline::CudaPipeline()
{
  cudaStreamCreate(&stream_);
}

CudaPipeline::~CudaPipeline()
{
  if (d_x_) cudaFree(d_x_);
  if (d_y_) cudaFree(d_y_);
  if (d_z_) cudaFree(d_z_);
  if (d_grid_max_zi_) cudaFree(d_grid_max_zi_);
  if (d_grid_z_) cudaFree(d_grid_z_);
  if (d_tmp_) cudaFree(d_tmp_);
  if (stream_) cudaStreamDestroy(stream_);
}

bool CudaPipeline::ensurePointCapacity(int n)
{
  if (n <= d_point_cap_) return true;

  if (d_x_) cudaFree(d_x_);
  if (d_y_) cudaFree(d_y_);
  if (d_z_) cudaFree(d_z_);

  if (!cudaOk(cudaMalloc(&d_x_, sizeof(float) * n))) return false;
  if (!cudaOk(cudaMalloc(&d_y_, sizeof(float) * n))) return false;
  if (!cudaOk(cudaMalloc(&d_z_, sizeof(float) * n))) return false;

  d_point_cap_ = n;
  return true;
}

bool CudaPipeline::ensureGridCapacity(int N)
{
  if (N <= d_grid_cap_) return true;

  if (d_grid_max_zi_) cudaFree(d_grid_max_zi_);
  if (d_grid_z_) cudaFree(d_grid_z_);
  if (d_tmp_) cudaFree(d_tmp_);

  if (!cudaOk(cudaMalloc(&d_grid_max_zi_, sizeof(int32_t) * N))) return false;
  if (!cudaOk(cudaMalloc(&d_grid_z_, sizeof(float) * N))) return false;
  if (!cudaOk(cudaMalloc(&d_tmp_, sizeof(float) * N))) return false;

  d_grid_cap_ = N;
  return true;
}

bool CudaPipeline::process(const float* h_x, const float* h_y, const float* h_z, int n,
                           const Params& p,
                           std::vector<float>& h_elevation)
{
  const int width  = p.width;
  const int height = p.height;
  const int N = width * height;
  if (width <= 0 || height <= 0 || n < 0) return false;

  h_elevation.assign(N, std::numeric_limits<float>::quiet_NaN());

  if (!ensurePointCapacity(n)) return false;
  if (!ensureGridCapacity(N)) return false;

  // copy points
  if (n > 0) {
    if (!cudaOk(cudaMemcpyAsync(d_x_, h_x, sizeof(float)*n, cudaMemcpyHostToDevice, stream_))) return false;
    if (!cudaOk(cudaMemcpyAsync(d_y_, h_y, sizeof(float)*n, cudaMemcpyHostToDevice, stream_))) return false;
    if (!cudaOk(cudaMemcpyAsync(d_z_, h_z, sizeof(float)*n, cudaMemcpyHostToDevice, stream_))) return false;

    int threads = 256;
    int blocks  = (n + threads - 1) / threads;
    pc2_gms_mask_points_by_z<<<blocks, threads, 0, stream_>>>(d_x_, d_y_, d_z_, n, p.z_min, p.z_max);
  }

  // empty sentinel (ordered(-inf))
  const int32_t empty_oi = floatToOrderedIntHost(-CUDART_INF_F);

  // init ordered-int grid
  {
    int threads = 256;
    int blocks  = (N + threads - 1) / threads;
    pc2_gms_init_int<<<blocks, threads, 0, stream_>>>(d_grid_max_zi_, N, empty_oi);
  }

  // existing kernel from gpu_gridmap.cu
  launch_points_to_grid_maxheight_orderedint(
      d_x_, d_y_, d_z_, n,
      p.origin_x, p.origin_y,
      p.resolution,
      width, height,
      d_grid_max_zi_,
      stream_);

  // ordered-int -> float
  {
    int threads = 256;
    int blocks  = (N + threads - 1) / threads;
    pc2_gms_orderedint_to_float<<<blocks, threads, 0, stream_>>>(d_grid_max_zi_, d_grid_z_, N, empty_oi);
  }

  // smoothing (ping-pong)
  const int iters = std::max(0, p.smooth_iterations);
  const int R     = std::max(0, p.smooth_radius_cells);

  if (iters > 0 && R > 0) {
    int threads = 128;
    int blocks  = (height + threads - 1) / threads;

    for (int k = 0; k < iters; ++k) {
      pc2_gms_smooth_box_nanmean<<<blocks, threads, 0, stream_>>>(d_grid_z_, d_tmp_, width, height, R);
      float* tmp = d_grid_z_;
      d_grid_z_ = d_tmp_;
      d_tmp_ = tmp;
    }
  }

  if (!cudaOk(cudaGetLastError())) return false;

  // copy back
  if (!cudaOk(cudaMemcpyAsync(h_elevation.data(), d_grid_z_, sizeof(float)*N, cudaMemcpyDeviceToHost, stream_))) return false;
  if (!cudaOk(cudaStreamSynchronize(stream_))) return false;

  return true;
}

} // namespace pc2_gridmap_smooth
