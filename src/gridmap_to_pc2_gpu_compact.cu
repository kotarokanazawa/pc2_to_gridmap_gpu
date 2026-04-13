// gridmap_to_pc2_gpu_compact.cu
// thrust-based compaction: keep only finite z

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

#include <vector>
#include <stdexcept>
#include <cmath>

struct IsFinite
{
  __host__ __device__ bool operator()(const float& z) const
  {
    // thrust side: use isfinite from <cmath> works in device with nvcc
    return isfinite(z);
  }
};

extern "C" void gpu_compact_xyz_isfinite(
    const float* h_x, const float* h_y, const float* h_z, int n,
    std::vector<float>& out_x, std::vector<float>& out_y, std::vector<float>& out_z)
{
  if (n <= 0) {
    out_x.clear(); out_y.clear(); out_z.clear();
    return;
  }

  try {
    thrust::device_vector<float> d_x(h_x, h_x + n);
    thrust::device_vector<float> d_y(h_y, h_y + n);
    thrust::device_vector<float> d_z(h_z, h_z + n);

    const int m = static_cast<int>(thrust::count_if(d_z.begin(), d_z.end(), IsFinite()));
    out_x.resize(m);
    out_y.resize(m);
    out_z.resize(m);

    thrust::device_vector<float> d_xo(m), d_yo(m), d_zo(m);

    auto in_begin = thrust::make_zip_iterator(thrust::make_tuple(d_x.begin(), d_y.begin(), d_z.begin()));
    auto in_end   = thrust::make_zip_iterator(thrust::make_tuple(d_x.end(),   d_y.end(),   d_z.end()));
    auto out_begin= thrust::make_zip_iterator(thrust::make_tuple(d_xo.begin(), d_yo.begin(), d_zo.begin()));

    thrust::copy_if(
        in_begin, in_end,
        d_z.begin(),          // stencil
        out_begin,
        IsFinite());

    thrust::copy(d_xo.begin(), d_xo.end(), out_x.begin());
    thrust::copy(d_yo.begin(), d_yo.end(), out_y.begin());
    thrust::copy(d_zo.begin(), d_zo.end(), out_z.begin());
  }
  catch (const std::exception& e) {
    throw std::runtime_error(std::string("thrust/cuda error: ") + e.what());
  }
}
