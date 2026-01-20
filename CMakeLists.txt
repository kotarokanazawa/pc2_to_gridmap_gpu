cmake_minimum_required(VERSION 3.0.2)
project(pc2_to_gridmap_gpu)

find_package(catkin REQUIRED COMPONENTS
  roscpp sensor_msgs grid_map_ros grid_map_msgs
)
# これだけ残す
set(CUDA_HOST_COMPILER /usr/bin/gcc-8)
find_package(CUDA REQUIRED)

# ここでは -ccbin を付けない
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3 --use_fast_math")


catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
)


# Build CUDA object
cuda_add_library(gpu_gridmap_cuda
  src/gpu_gridmap.cu
)

add_executable(pc2_to_gridmap_gpu_node
  src/pc2_to_gridmap_gpu_node.cpp
)

target_link_libraries(pc2_to_gridmap_gpu_node
  gpu_gridmap_cuda
  ${catkin_LIBRARIES}
  ${CUDA_LIBRARIES}
)

add_dependencies(pc2_to_gridmap_gpu_node
  ${catkin_EXPORTED_TARGETS}
)