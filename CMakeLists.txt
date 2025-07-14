cmake_minimum_required(VERSION 3.21)
project(deeplearningLA CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2")
find_package(OpenMP REQUIRED) # Find OpenMP package

# Don't declare CUDA as required here
# find_package(CUDA QUIET)
# if(CUDA_FOUND)
#     set(CMAKE_CUDA_STANDARD 17)
#     set(CMAKE_CUDA_STANDARD_REQUIRED ON)
#     message(STATUS "CUDA found: enabling CUDA support")
#     enable_language(CUDA)
#     set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
#     add_definitions(-DCUDA_ENABLED)
# else()
#     message(WARNING "CUDA not found: building CPU-only fallback")
# endif()


add_subdirectory(core)

add_subdirectory(test)

add_custom_target(
    src
    DEPENDS kernel graph framework las gpu
)

add_custom_target(
    test
    DEPENDS kernel graph framework las gpu
)

set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/build/)
install(TARGETS kernel
    graph
    framework
    las
    LIBRARY DESTINATION lib)
