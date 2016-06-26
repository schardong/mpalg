cmake_minimum_required(VERSION 2.8)

find_package(OpenCV REQUIRED)

add_library(mpalg mds.cpp lamp.cpp)

target_include_directories (mpalg PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
