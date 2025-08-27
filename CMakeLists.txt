cmake_minimum_required(VERSION 3.21)
project(deeplearningLA CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2")
find_package(OpenMP REQUIRED) # Find OpenMP package

# Prefer relative rpaths (so executables find your .so’s next to them)
set(CMAKE_BUILD_RPATH_USE_ORIGIN ON)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# Let installed/runnable binaries find project libs in ../lib and alongside the exe
set(CMAKE_BUILD_RPATH "$ORIGIN;$ORIGIN/../lib")
set(CMAKE_INSTALL_RPATH "$ORIGIN;$ORIGIN/../lib")

# Append system libstdc++/glibc dirs (adjust versions/paths for your distro)
list(APPEND CMAKE_BUILD_RPATH "/usr/lib/gcc/x86_64-linux-gnu/13" "/usr/lib/x86_64-linux-gnu")
list(APPEND CMAKE_INSTALL_RPATH "/usr/lib/gcc/x86_64-linux-gnu/13" "/usr/lib/x86_64-linux-gnu")



option(ENABLE_CUDA "Build with CUDA support" ON)
# Options to allow system libs if you want later
option(USE_SYSTEM_GTEST "Use system-installed GoogleTest" OFF)
option(USE_SYSTEM_BENCHMARK "Use system-installed Google Benchmark" OFF)


add_subdirectory(thirdparty)

add_subdirectory(core)

add_subdirectory(api)

add_subdirectory(test)

# add_custom_target(
#     src
#     DEPENDS core tensor_api
# )

# add_custom_target(
#     test
#     DEPENDS core tensor_api
# )

set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/build/)
install(TARGETS core
    tensor_api
    LIBRARY DESTINATION lib)
