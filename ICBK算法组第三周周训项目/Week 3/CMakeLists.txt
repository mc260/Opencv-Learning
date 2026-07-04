cmake_minimum_required(VERSION 3.17)
project(Week_3)

set(CMAKE_CXX_STANDARD 14)

# 寻找OpenCV包
find_package(OpenCV REQUIRED)
# 包含OpenCV的头文件目录
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(Week_3 IO.cpp
        main.cpp
        no1.h)
# 将OpenCV库链接到你的项目
target_link_libraries(Week_3 ${OpenCV_LIBS})