cmake_minimum_required(VERSION 2.8)

find_package(OpenCV REQUIRED)

add_executable(t.mpalg main.cpp)
target_link_libraries(t.mpalg ${OpenCV_LIBS})
target_link_libraries (t.mpalg LINK_PUBLIC mpalg)
