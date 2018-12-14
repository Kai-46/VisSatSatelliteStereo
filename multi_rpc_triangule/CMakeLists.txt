cmake_minimum_required(VERSION 2.8)

project(multi_rpc_triangulate)

find_package(Ceres REQUIRED)

include_directories(${CERES_INCLUDE_DIRS})

add_executable(multi_rpc_triangulate multi_rpc_triangulate.cpp)
target_link_libraries(multi_rpc_triangulate ${CERES_LIBRARIES})
