cmake_minimum_required(VERSION 3.0.2)
project(pc2_to_gridmap_gpu)

find_package(catkin REQUIRED COMPONENTS
  roscpp sensor_msgs grid_map_ros grid_map_msgs
)

find_program(CUDA_GCC8 gcc-8)
find_program(CUDA_GXX8 g++-8)
if(NOT CUDA_GCC8 OR NOT CUDA_GXX8)
  message(FATAL_ERROR "gcc-8 and g++-8 are required to build pc2_to_gridmap_gpu with CUDA 10.1")
endif()

set(CUDA_HOST_COMPILER ${CUDA_GXX8})
find_package(CUDA REQUIRED)

# ここでは -ccbin を付けない
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3 --use_fast_math")
message(STATUS "pc2_to_gridmap_gpu CUDA host compiler: ${CUDA_HOST_COMPILER}")
message(STATUS "pc2_to_gridmap_gpu CUDA C++ host compiler: ${CUDA_GXX8}")


catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
)


# Build CUDA object
cuda_add_library(gpu_gridmap_cuda
  src/gpu_gridmap.cu
  src/gridmap_to_pc2_gpu_compact.cu
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


add_executable(gridmap_to_pc2_gpu_node
  src/gridmap_to_pc2_gpu_node.cpp
)

target_link_libraries(gridmap_to_pc2_gpu_node
  gpu_gridmap_cuda
  ${catkin_LIBRARIES}
  ${CUDA_LIBRARIES}
)

add_dependencies(gridmap_to_pc2_gpu_node
  ${catkin_EXPORTED_TARGETS}
)
