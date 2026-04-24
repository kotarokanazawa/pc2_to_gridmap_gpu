// pc2_to_gridmap_gpu_node.cpp (ROS1 Noetic + CUDA)
// PointCloud2(XYZ) -> GridMap("elevation") with per-cell MAX height (zmax) on GPU
// - TF: point cloud frame -> map_frame
// - GPU aggregation: ordered-int + atomicMax (works with negative z)
// - Invalid (NaN/Inf) points are ignored
//
// params:
//  ~input_topic (default: /points)
//  ~map_frame   (default: map)
//  ~resolution  (default: 0.05)
//  ~length_x    (default: 10.0)
//  ~length_y    (default: 10.0)
//  ~origin_x    (default: -5.0)  // lower-left corner in map_frame
//  ~origin_y    (default: -5.0)

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>

#include <grid_map_msgs/GridMap.h>
#include <grid_map_ros/grid_map_ros.hpp>
#include <grid_map_core/grid_map_core.hpp>

#include <cuda_runtime.h>

#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2/LinearMath/Transform.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

#include <vector>
#include <limits>
#include <cstring>
#include <stdexcept>
#include <algorithm>
#include <cmath>

// CUDA launcher (from .cu)
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

// read float from PointCloud2 at offset
static inline float readFloat(const uint8_t* ptr)
{
  float v;
  std::memcpy(&v, ptr, sizeof(float));
  return v;
}

// ordered-int transform (CPU side)
static inline int floatToOrderedInt_host(float f)
{
  int i;
  std::memcpy(&i, &f, sizeof(float));
  return (i >= 0) ? i : (i ^ 0x7fffffff);
}
static inline float orderedIntToFloat_host(int i)
{
  int j = (i >= 0) ? i : (i ^ 0x7fffffff);
  float f;
  std::memcpy(&f, &j, sizeof(float));
  return f;
}

class Pc2ToGridmapGpuMax
{
public:
  Pc2ToGridmapGpuMax(ros::NodeHandle& nh, ros::NodeHandle& pnh)
  : tf_listener_(tf_buffer_)
  {
    pnh.param<std::string>("input_topic", input_topic_, std::string("/points"));
    pnh.param<std::string>("map_frame",   map_frame_,   std::string("map"));
    pnh.param<double>("resolution", resolution_, 0.05);
    pnh.param<double>("length_x",   length_x_,   10.0);
    pnh.param<double>("length_y",   length_y_,   10.0);

    // lower-left origin in map_frame
    pnh.param<double>("origin_x", origin_x_, -5.0);
    pnh.param<double>("origin_y", origin_y_, -5.0);

    width_  = static_cast<int>(std::round(length_x_ / resolution_));
    height_ = static_cast<int>(std::round(length_y_ / resolution_));
    if (width_ <= 0 || height_ <= 0) throw std::runtime_error("Invalid grid size.");

    pub_ = nh.advertise<grid_map_msgs::GridMap>("elevation_grid_map", 1);
    sub_ = nh.subscribe(input_topic_, 1, &Pc2ToGridmapGpuMax::cb, this);

    cudaStreamCreate(&stream_);

    // GPU grid buffer (ordered-int)
    cudaMalloc(&d_grid_max_zi_, sizeof(int) * width_ * height_);

    // init pattern: -INF (for max)
    const float ninf = -std::numeric_limits<float>::infinity();
    const int ninf_i = floatToOrderedInt_host(ninf);
    h_grid_init_zi_.assign(width_ * height_, ninf_i);
    h_grid_out_zi_.assign(width_ * height_, ninf_i);

    ROS_INFO("pc2_to_gridmap_gpu_max: %dx%d res=%.3f origin_ll=(%.3f,%.3f) map_frame=%s",
             width_, height_, resolution_, origin_x_, origin_y_, map_frame_.c_str());
  }

  ~Pc2ToGridmapGpuMax()
  {
    if (d_x_) cudaFree(d_x_);
    if (d_y_) cudaFree(d_y_);
    if (d_z_) cudaFree(d_z_);
    if (d_grid_max_zi_) cudaFree(d_grid_max_zi_);
    cudaStreamDestroy(stream_);
  }

private:
  void cb(const sensor_msgs::PointCloud2ConstPtr& msg)
  {
    // Locate x,y,z offsets
    int off_x = -1, off_y = -1, off_z = -1;
    for (const auto& f : msg->fields) {
      if (f.name == "x") off_x = f.offset;
      else if (f.name == "y") off_y = f.offset;
      else if (f.name == "z") off_z = f.offset;
    }
    if (off_x < 0 || off_y < 0 || off_z < 0) {
      ROS_WARN_THROTTLE(1.0, "PointCloud2 has no x/y/z fields.");
      return;
    }
    if (msg->point_step < (uint32_t)std::max({off_x, off_y, off_z}) + 4) {
      ROS_WARN_THROTTLE(1.0, "PointCloud2 point_step too small.");
      return;
    }

    const int n = static_cast<int>(msg->width * msg->height);
    if (n <= 0) return;

    // TF: cloud frame -> map_frame
    tf2::Transform tf_msg_to_map;
    if (msg->header.frame_id == map_frame_) {
      tf_msg_to_map.setIdentity();
    } else {
      geometry_msgs::TransformStamped T;
      try {
        T = tf_buffer_.lookupTransform(
            map_frame_, msg->header.frame_id, msg->header.stamp, ros::Duration(0.05));
      } catch (const tf2::TransformException& e) {
        ROS_WARN_THROTTLE(1.0, "TF lookup failed (%s -> %s): %s",
                          msg->header.frame_id.c_str(), map_frame_.c_str(), e.what());
        return;
      }
      tf2::fromMsg(T.transform, tf_msg_to_map);
    }

    // Host arrays (XYZ)
    h_x_.resize(n);
    h_y_.resize(n);
    h_z_.resize(n);

    const uint8_t* data = msg->data.data();
    const uint32_t step = msg->point_step;

    // Extract XYZ (CPU) + TF transform to map_frame
    for (int i = 0; i < n; ++i) {
      const uint8_t* p = data + i * step;

      const float x = readFloat(p + off_x);
      const float y = readFloat(p + off_y);
      const float z = readFloat(p + off_z);

      const tf2::Vector3 v_map = tf_msg_to_map * tf2::Vector3(x, y, z);

      h_x_[i] = static_cast<float>(v_map.x());
      h_y_[i] = static_cast<float>(v_map.y());
      h_z_[i] = static_cast<float>(v_map.z());
    }

    ensureDeviceCapacity(n);

    // init grid to -INF
    cudaMemcpyAsync(d_grid_max_zi_, h_grid_init_zi_.data(),
                    sizeof(int) * h_grid_init_zi_.size(),
                    cudaMemcpyHostToDevice, stream_);

    // Copy XYZ to GPU
    cudaMemcpyAsync(d_x_, h_x_.data(), sizeof(float) * n, cudaMemcpyHostToDevice, stream_);
    cudaMemcpyAsync(d_y_, h_y_.data(), sizeof(float) * n, cudaMemcpyHostToDevice, stream_);
    cudaMemcpyAsync(d_z_, h_z_.data(), sizeof(float) * n, cudaMemcpyHostToDevice, stream_);

    // Launch kernel (MAX height)
    launch_points_to_grid_maxheight_orderedint(
        d_x_, d_y_, d_z_, n,
        (float)origin_x_, (float)origin_y_,
        (float)resolution_,
        width_, height_,
        d_grid_max_zi_, stream_);

    // Copy back grid (ordered int)
    cudaMemcpyAsync(h_grid_out_zi_.data(), d_grid_max_zi_,
                    sizeof(int) * h_grid_out_zi_.size(),
                    cudaMemcpyDeviceToHost, stream_);

    cudaStreamSynchronize(stream_);

    // Build GridMap (CPU)
    grid_map::GridMap gm;
    gm.setFrameId(map_frame_);
    gm.setGeometry(grid_map::Length(length_x_, length_y_),
                   resolution_,
                   grid_map::Position(origin_x_ + length_x_ * 0.5,
                                      origin_y_ + length_y_ * 0.5));
    gm.add("elevation", NAN);

    // Convert ordered int -> float, and write to grid_map by metric position.
    // grid_map's matrix indices are not the same convention as the CUDA
    // lower-left ix/iy grid, so avoid direct matrix indexing here.
    const float ninf = -std::numeric_limits<float>::infinity();
    for (int iy = 0; iy < height_; ++iy) {
      for (int ix = 0; ix < width_; ++ix) {
        const int zi = h_grid_out_zi_[iy * width_ + ix];
        const float zmax = orderedIntToFloat_host(zi);
        if (!std::isfinite(zmax) || zmax == ninf) {
          continue;
        }

        const double x = origin_x_ + (static_cast<double>(ix) + 0.5) * resolution_;
        const double y = origin_y_ + (static_cast<double>(iy) + 0.5) * resolution_;
        const grid_map::Position position(x, y);
        grid_map::Index index;
        if (gm.getIndex(position, index)) {
          gm.at("elevation", index) = zmax;
        }
      }
    }

    grid_map_msgs::GridMap out;
    grid_map::GridMapRosConverter::toMessage(gm, out);
    out.info.header.stamp = msg->header.stamp;
    out.info.header.frame_id = map_frame_;
    pub_.publish(out);
  }

  void ensureDeviceCapacity(int n)
  {
    if (n <= d_capacity_) return;

    if (d_x_) cudaFree(d_x_);
    if (d_y_) cudaFree(d_y_);
    if (d_z_) cudaFree(d_z_);

    cudaMalloc(&d_x_, sizeof(float) * n);
    cudaMalloc(&d_y_, sizeof(float) * n);
    cudaMalloc(&d_z_, sizeof(float) * n);
    d_capacity_ = n;
  }

private:
  ros::Subscriber sub_;
  ros::Publisher pub_;

  std::string input_topic_;
  std::string map_frame_;

  double resolution_{0.05};
  double length_x_{10.0}, length_y_{10.0};
  double origin_x_{-5.0}, origin_y_{-5.0};
  int width_{0}, height_{0};

  // TF
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;

  // Host buffers
  std::vector<float> h_x_, h_y_, h_z_;
  std::vector<int>   h_grid_init_zi_;
  std::vector<int>   h_grid_out_zi_;

  // Device buffers
  float* d_x_{nullptr};
  float* d_y_{nullptr};
  float* d_z_{nullptr};
  int d_capacity_{0};

  int* d_grid_max_zi_{nullptr};

  cudaStream_t stream_{};
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "pc2_to_gridmap_gpu_max");
  ros::NodeHandle nh, pnh("~");
  try {
    Pc2ToGridmapGpuMax node(nh, pnh);
    ros::spin();
  } catch (const std::exception& e) {
    ROS_ERROR("Fatal: %s", e.what());
  }
  return 0;
}
