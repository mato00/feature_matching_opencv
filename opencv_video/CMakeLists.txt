cmake_minimum_required( VERSION 2.8 )
project( realtime_feature_match )

# set(CMAKE_BUILD_TYPE "Debug")
# set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb")

# 添加c++ 11标准支持
set( CMAKE_CXX_FLAGS "-std=c++11" )

# 寻找OpenCV库
set(OpenCV_DIR /usr/local/share/OpenCV)
find_package( OpenCV 3.2.0 REQUIRED )
# 添加头文件
# include_directories("/usr/local/include")
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(realtime_feature_match realtime_feature_match.cpp)
# 链接OpenCV库
# link_directories(/usr/local/lib)
target_link_libraries(realtime_feature_match ${OpenCV_LIBS})
