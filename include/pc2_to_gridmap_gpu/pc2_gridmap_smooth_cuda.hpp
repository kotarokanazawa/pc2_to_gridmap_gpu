#pragma once
#include <cuda_runtime.h>
#include <vector>
#include <cstdint>

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

struct Params
{
  // grid geometry (origin is LOWER-LEFT)
  float origin_x = -5.0f;
  float origin_y = -5.0f;
  float resolution = 0.05f;
  int width = 200;
  int height = 200;

  // point z filter
  float z_min = -5.0f;
  float z_max =  5.0f;

  // smoothing
  int smooth_iterations = 3;
  int smooth_radius_cells = 2;
};

class CudaPipeline
{
public:
  CudaPipeline();
  ~CudaPipeline();

  // process a single cloud (SoA host arrays) -> elevation (size width*height)
  // output uses NaN for empty cells
  bool process(const float* h_x, const float* h_y, const float* h_z, int n,
               const Params& p,
               std::vector<float>& h_elevation);

private:
  // non-copyable
  CudaPipeline(const CudaPipeline&) = delete;
  CudaPipeline& operator=(const CudaPipeline&) = delete;

  bool ensurePointCapacity(int n);
  bool ensureGridCapacity(int N);

private:
  cudaStream_t stream_{};

  // device point buffers
  float* d_x_{nullptr};
  float* d_y_{nullptr};
  float* d_z_{nullptr};
  int d_point_cap_{0};

  // device grids
  int*   d_grid_max_zi_{nullptr};  // ordered-int zmax
  float* d_grid_z_{nullptr};       // float elevation (NaN for empty)
  float* d_tmp_{nullptr};          // smoothing temp
  int d_grid_cap_{0};
};

} // namespace pc2_gridmap_smooth
