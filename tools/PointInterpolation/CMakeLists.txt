cmake_minimum_required(VERSION 3.1)
project(pointInter)

set(CMAKE_CXX_FLAGS "-std=c++14")

set(CMAKE_BUILD_TYPE release)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")


find_package(OpenMP REQUIRED)
find_package(OpenCV REQUIRED)
find_package(PCL 1.10 REQUIRED)

add_executable(pointInterHaomo src/pointInterHaomo.cpp) 
add_executable(pointInterKitti src/pointInterKitti.cpp) 



target_link_libraries(pointInterHaomo ${PCL_LIBRARIES} ${OpenCV_LIBS} OpenMP::OpenMP_CXX)
target_link_libraries(pointInterKitti ${PCL_LIBRARIES} ${OpenCV_LIBS} OpenMP::OpenMP_CXX)

