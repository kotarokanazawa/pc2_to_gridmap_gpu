#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>

#include <grid_map_core/grid_map_core.hpp>
#include <grid_map_ros/grid_map_ros.hpp>
#include <grid_map_msgs/GridMap.h>

#include <vector>
#include <string>
#include <limits>
#include <cstring>
#include <cmath>
#include <algorithm>

#include "pc2_to_gridmap_gpu/pc2_gridmap_smooth_cuda.hpp"

static inline float readFloat(const uint8_t* ptr)
{
  float v;
  std::memcpy(&v, ptr, sizeof(float));
  return v;
}

class Pc2GridmapSmoothNode
{
public:
  Pc2GridmapSmoothNode(ros::NodeHandle& nh, ros::NodeHandle& pnh)
  {
    pnh.param<std::string>("input_topic",  input_topic_,  std::string("/points"));
    pnh.param<std::string>("output_topic", output_topic_, std::string("/elevation_grid_map"));
    pnh.param<std::string>("map_frame",    map_frame_,    std::string("map"));

    pnh.param<double>("resolution", resolution_, 0.05);
    pnh.param<double>("length_x",   length_x_,   10.0);
    pnh.param<double>("length_y",   length_y_,   10.0);

    // IMPORTANT: lower-left origin for CUDA grid index mapping
    pnh.param<double>("origin_x", origin_x_, -5.0); // lower-left x
    pnh.param<double>("origin_y", origin_y_, -5.0); // lower-left y

    pnh.param<double>("z_min", z_min_, -5.0);
    pnh.param<double>("z_max", z_max_,  5.0);

    pnh.param<int>("smooth_iterations",   smooth_iterations_,   3);
    pnh.param<int>("smooth_radius_cells", smooth_radius_cells_, 2);

    width_  = std::max(1, (int)std::round(length_x_ / resolution_));
    height_ = std::max(1, (int)std::round(length_y_ / resolution_));

    sub_ = nh.subscribe(input_topic_, 1, &Pc2GridmapSmoothNode::cb, this);
    pub_ = nh.advertise<grid_map_msgs::GridMap>(output_topic_, 1, true);

    ROS_INFO("pc2_gridmap_smooth: input=%s output=%s grid=%dx%d res=%.3f origin_ll=(%.3f,%.3f)",
             input_topic_.c_str(), output_topic_.c_str(),
             width_, height_, resolution_, origin_x_, origin_y_);
  }

private:
  void cb(const sensor_msgs::PointCloud2ConstPtr& msg)
  {
    // locate xyz offsets
    int off_x=-1, off_y=-1, off_z=-1;
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

    const int n = (int)(msg->width * msg->height);
    if (n <= 0) return;

    // unpack XYZ (CPU)
    h_x_.resize(n);
    h_y_.resize(n);
    h_z_.resize(n);

    const uint8_t* data = msg->data.data();
    const uint32_t step = msg->point_step;

    for (int i = 0; i < n; ++i) {
      const uint8_t* p = data + (size_t)i * step;
      h_x_[i] = readFloat(p + off_x);
      h_y_[i] = readFloat(p + off_y);
      h_z_[i] = readFloat(p + off_z);
    }

    // run GPU pipeline
    pc2_gridmap_smooth::Params gp;
    gp.origin_x = (float)origin_x_;
    gp.origin_y = (float)origin_y_;
    gp.resolution = (float)resolution_;
    gp.width  = width_;
    gp.height = height_;
    gp.z_min = (float)z_min_;
    gp.z_max = (float)z_max_;
    gp.smooth_iterations = smooth_iterations_;
    gp.smooth_radius_cells = smooth_radius_cells_;

    std::vector<float> elev;
    if (!cuda_.process(h_x_.data(), h_y_.data(), h_z_.data(), n, gp, elev)) {
      ROS_WARN_THROTTLE(1.0, "CUDA process failed");
      return;
    }

    // Build GridMap
    grid_map::GridMap map;
    map.setFrameId(map_frame_);

    // grid_map geometry uses CENTER position.
    const double center_x = origin_x_ + length_x_ * 0.5;
    const double center_y = origin_y_ + length_y_ * 0.5;

    map.setGeometry(grid_map::Length(length_x_, length_y_), resolution_,
                    grid_map::Position(center_x, center_y));

    const std::string layer = "elevation";
    map.add(layer, std::numeric_limits<float>::quiet_NaN());
    auto& E = map[layer];

    for (int r = 0; r < height_; ++r) {
      for (int c = 0; c < width_; ++c) {
        E(r, c) = elev[r * width_ + c];
      }
    }

    grid_map_msgs::GridMap out;
    grid_map::GridMapRosConverter::toMessage(map, out);
    out.info.header.stamp = msg->header.stamp;
    out.info.header.frame_id = map_frame_;
    pub_.publish(out);
  }

private:
  ros::Subscriber sub_;
  ros::Publisher  pub_;

  std::string input_topic_;
  std::string output_topic_;
  std::string map_frame_;

  double resolution_{0.05};
  double length_x_{10.0}, length_y_{10.0};
  double origin_x_{-5.0}, origin_y_{-5.0}; // LOWER-LEFT
  double z_min_{-5.0}, z_max_{5.0};
  int smooth_iterations_{3};
  int smooth_radius_cells_{2};

  int width_{0}, height_{0};

  std::vector<float> h_x_, h_y_, h_z_;
  pc2_gridmap_smooth::CudaPipeline cuda_;
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "pc2_gridmap_smooth");
  ros::NodeHandle nh;
  ros::NodeHandle pnh("~");

  Pc2GridmapSmoothNode node(nh, pnh);
  ros::spin();
  return 0;
}
