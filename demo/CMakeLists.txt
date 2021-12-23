cmake_minimum_required(VERSION 3.10)
project(demo)


set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

INCLUDE_DIRECTORIES(./libtorch/include ./libtorch/include/torch/csrc/api/include)
set(Torch_DIR ./libtorch/share/cmake/Torch/)
find_package(Torch REQUIRED)

# set(OpenCV_DIR /root/miniconda3/envs/python37/include/)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})




add_executable(demo main.cpp)
target_link_libraries(demo ${TORCH_LIBRARIES} ${OpenCV_LIBS})
set_property(TARGET demo PROPERTY CXX_STANDARD 14)