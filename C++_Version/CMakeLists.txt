cmake_minimum_required(VERSION 4.0)
project(C___Version)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
set(CMAKE_CXX_STANDARD 20)

add_executable(C___Version main.cpp)
target_link_libraries(C___Version ${OpenCV_LIBS})