cmake_minimum_required(VERSION 3.21)
project(deeplearningLA CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2")
find_package(OpenMP REQUIRED) # Find OpenMP package

enable_testing()

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


# ---------------- GoogleTest ----------------
if(USE_SYSTEM_GTEST)
  find_package(GTest REQUIRED)
else()
  include(FetchContent)
  set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.15.2
  )
  FetchContent_MakeAvailable(googletest)
endif()


# ---------------- Google Benchmark ----------------
if(USE_SYSTEM_BENCHMARK)
  find_package(benchmark REQUIRED)
else()
  include(FetchContent)
  set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
  set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
  set(BENCHMARK_ENABLE_WERROR OFF CACHE BOOL "" FORCE)
  set(BENCHMARK_DOWNLOAD_DEPENDENCIES OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(
    benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.8.5
  )

  set(BENCHMARK_ENABLE_TESTING OFF)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
  FetchContent_MakeAvailable(benchmark)
endif()


# ----------------- ABSL Logging --------------------
set(BUILD_SHARED_LIBS ON)
include(FetchContent)
FetchContent_Declare(
  abseil
  GIT_REPOSITORY https://github.com/abseil/abseil-cpp.git
  GIT_TAG 20240722.0 # example: stable release
)
FetchContent_MakeAvailable(abseil)

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
