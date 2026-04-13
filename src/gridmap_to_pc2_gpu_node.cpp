// gridmap_to_pc2_gpu_node.cpp (ROS1 Noetic + CUDA thrust)
// Subscribe: grid_map_msgs/GridMap
// Publish:   sensor_msgs/PointCloud2 (XYZ), one point per valid cell center
// GPU: compact (remove NaN/Inf) using thrust::copy_if on device
//
// params:
//  ~input_topic   (default: /grid_map)
//  ~output_topic  (default: /grid_map_points)
//  ~layer         (default: elevation)
//  ~frame_id      (default: "" -> use incoming GridMap frame)
//  ~queue_size    (default: 1)

#include <ros/ros.h>
#include <grid_map_msgs/GridMap.h>
#include <grid_map_ros/grid_map_ros.hpp>
#include <grid_map_core/grid_map_core.hpp>

#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>

#include <cuda_runtime.h>

#include <vector>
#include <string>
#include <stdexcept>
#include <cmath>

// implemented in .cu
extern "C" void gpu_compact_xyz_isfinite(
    const float* h_x, const float* h_y, const float* h_z, int n,
    std::vector<float>& out_x, std::vector<float>& out_y, std::vector<float>& out_z);

class GridmapToPc2Gpu
{
public:
  GridmapToPc2Gpu(ros::NodeHandle& nh, ros::NodeHandle& pnh)
  {
    pnh.param<std::string>("input_topic",  input_topic_,  std::string("/grid_map"));
    pnh.param<std::string>("output_topic", output_topic_, std::string("/grid_map_points"));
    pnh.param<std::string>("layer",        layer_,        std::string("elevation"));
    pnh.param<std::string>("frame_id",     frame_id_,     std::string(""));
    pnh.param<int>("queue_size", queue_size_, 1);

    pub_ = nh.advertise<sensor_msgs::PointCloud2>(output_topic_, 1);
    sub_ = nh.subscribe(input_topic_, queue_size_, &GridmapToPc2Gpu::cb, this);

    ROS_INFO("gridmap_to_pc2_gpu: sub=%s pub=%s layer=%s",
             input_topic_.c_str(), output_topic_.c_str(), layer_.c_str());
  }

private:
  void cb(const grid_map_msgs::GridMapConstPtr& msg)
  {
    grid_map::GridMap gm;
    if (!grid_map::GridMapRosConverter::fromMessage(*msg, gm)) {
      ROS_WARN_THROTTLE(1.0, "fromMessage failed.");
      return;
    }
    if (!gm.exists(layer_)) {
      ROS_WARN_THROTTLE(1.0, "Layer '%s' not found.", layer_.c_str());
      return;
    }

    const int size_x = static_cast<int>(gm.getSize()(0));
    const int size_y = static_cast<int>(gm.getSize()(1));
    const int n = size_x * size_y;
    if (n <= 0) return;

    // Build full (x,y,z) arrays on CPU with correct grid_map indexing / circular buffer
    h_x_.clear(); h_y_.clear(); h_z_.clear();
    h_x_.reserve(n); h_y_.reserve(n); h_z_.reserve(n);

    for (grid_map::GridMapIterator it(gm); !it.isPastEnd(); ++it) {
      const grid_map::Index idx(*it);
      grid_map::Position pos;
      gm.getPosition(idx, pos);

      const float z = gm.at(layer_, idx);
      h_x_.push_back(static_cast<float>(pos.x()));
      h_y_.push_back(static_cast<float>(pos.y()));
      h_z_.push_back(z);
    }

    // GPU compact: remove NaN/Inf based on z
    std::vector<float> x_ok, y_ok, z_ok;
    try {
      gpu_compact_xyz_isfinite(
          h_x_.data(), h_y_.data(), h_z_.data(), static_cast<int>(h_z_.size()),
          x_ok, y_ok, z_ok);
    } catch (const std::exception& e) {
      ROS_WARN_THROTTLE(1.0, "GPU compact failed: %s", e.what());
      return;
    }

    const size_t m = z_ok.size();
    if (m == 0) {
      // publish empty cloud (still useful for sync)
      sensor_msgs::PointCloud2 out;
      out.header.stamp = msg->info.header.stamp;
      out.header.frame_id = (frame_id_.empty() ? gm.getFrameId() : frame_id_);
      out.height = 1;
      out.width  = 0;
      out.is_bigendian = false;
      out.is_dense = true;
      pub_.publish(out);
      return;
    }

    // Build PointCloud2 XYZ
    sensor_msgs::PointCloud2 out;
    out.header.stamp = msg->info.header.stamp;
    out.header.frame_id = (frame_id_.empty() ? gm.getFrameId() : frame_id_);
    out.height = 1;
    out.width  = static_cast<uint32_t>(m);
    out.is_bigendian = false;
    out.is_dense = true;

    sensor_msgs::PointCloud2Modifier mod(out);
    mod.setPointCloud2FieldsByString(1, "xyz");
    mod.resize(m);

    sensor_msgs::PointCloud2Iterator<float> it_x(out, "x");
    sensor_msgs::PointCloud2Iterator<float> it_y(out, "y");
    sensor_msgs::PointCloud2Iterator<float> it_z(out, "z");
    for (size_t i = 0; i < m; ++i, ++it_x, ++it_y, ++it_z) {
      *it_x = x_ok[i];
      *it_y = y_ok[i];
      *it_z = z_ok[i];
    }

    pub_.publish(out);
  }

private:
  ros::Subscriber sub_;
  ros::Publisher  pub_;

  std::string input_topic_;
  std::string output_topic_;
  std::string layer_;
  std::string frame_id_;
  int queue_size_{1};

  std::vector<float> h_x_, h_y_, h_z_;
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "gridmap_to_pc2_gpu");
  ros::NodeHandle nh, pnh("~");
  try {
    GridmapToPc2Gpu node(nh, pnh);
    ros::spin();
  } catch (const std::exception& e) {
    ROS_ERROR("Fatal: %s", e.what());
  }
  return 0;
}
