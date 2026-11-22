cmake_minimum_required(VERSION 3.21)
project(learn_deep_custom LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# `custom/` is deliberately NOT part of the framework build. It is configured
# separately, after the framework libraries exist, and consumes only those
# binaries plus the public <api/tensor.h> header (the test below does exactly
# that). See README.md for the current limitations of that boundary.
set(LD_FRAMEWORK_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." CACHE PATH
    "Root of the LearnDeep source tree")
set(LD_FRAMEWORK_BUILD_DIR "${LD_FRAMEWORK_ROOT}/build/debug" CACHE PATH
    "Directory containing the built framework libraries")
get_filename_component(LD_FRAMEWORK_ROOT "${LD_FRAMEWORK_ROOT}" ABSOLUTE)
get_filename_component(LD_FRAMEWORK_BUILD_DIR "${LD_FRAMEWORK_BUILD_DIR}" ABSOLUTE)

# The public header tree. <api/tensor.h> resolves here, and it transitively
# includes the core/loss/optimizer headers it needs.
set(LD_FRAMEWORK_INCLUDE_DIR "${LD_FRAMEWORK_ROOT}/LearnDeep")

find_library(LD_TENSOR_API_LIBRARY tensor_api
             PATHS "${LD_FRAMEWORK_BUILD_DIR}/LearnDeep/api"
                   "${LD_FRAMEWORK_BUILD_DIR}"
             REQUIRED)
find_library(LD_LAYER_LIBRARY layer
             PATHS "${LD_FRAMEWORK_BUILD_DIR}/LearnDeep/layers"
                   "${LD_FRAMEWORK_BUILD_DIR}"
             REQUIRED)
find_library(LD_CORE_LIBRARY core
             PATHS "${LD_FRAMEWORK_BUILD_DIR}/LearnDeep/core"
                   "${LD_FRAMEWORK_BUILD_DIR}"
             REQUIRED)
find_library(LD_LOSS_LIBRARY loss
             PATHS "${LD_FRAMEWORK_BUILD_DIR}/LearnDeep/losses"
                   "${LD_FRAMEWORK_BUILD_DIR}"
             REQUIRED)

set(LD_FRAMEWORK_LIBRARIES
    "${LD_TENSOR_API_LIBRARY}"
    "${LD_LAYER_LIBRARY}"
    "${LD_CORE_LIBRARY}"
    "${LD_LOSS_LIBRARY}")

# ---------------------------------------------------------------------------
# layer/ -- self-registering custom layer plugin
# ---------------------------------------------------------------------------
file(GLOB CUSTOM_LAYER_SOURCES CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/layer/*.cpp")

if(CUSTOM_LAYER_SOURCES)
  add_library(custom_layer SHARED ${CUSTOM_LAYER_SOURCES})
  target_include_directories(custom_layer PRIVATE "${LD_FRAMEWORK_INCLUDE_DIR}")
  target_link_libraries(custom_layer PRIVATE ${LD_FRAMEWORK_LIBRARIES})
endif()

# ---------------------------------------------------------------------------
# loss/ -- self-registering custom loss plugin
# ---------------------------------------------------------------------------
file(GLOB CUSTOM_LOSS_SOURCES CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/loss/*.cpp")

if(CUSTOM_LOSS_SOURCES)
  add_library(custom_loss SHARED ${CUSTOM_LOSS_SOURCES})
  target_include_directories(custom_loss PRIVATE "${LD_FRAMEWORK_INCLUDE_DIR}")
  target_link_libraries(custom_loss PRIVATE ${LD_FRAMEWORK_LIBRARIES})
endif()

# optimizer/ -- a future experiment area can be added the same way, without
# touching the framework CMake files.

# ---------------------------------------------------------------------------
# test/ -- standalone checks that consume only the public API header + libs
# ---------------------------------------------------------------------------
enable_testing()

file(GLOB CUSTOM_TEST_SOURCES CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/test/*.cpp")

# One executable per test source: each test is an independent out-of-framework
# consumer with its own main(), so adding a file here needs no CMake edit.
foreach(test_source IN LISTS CUSTOM_TEST_SOURCES)
  get_filename_component(test_name "${test_source}" NAME_WE)
  add_executable(${test_name} "${test_source}")
  target_include_directories(${test_name} PRIVATE "${LD_FRAMEWORK_INCLUDE_DIR}")
  target_compile_definitions(${test_name} PRIVATE
    LD_CUSTOM_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test/data")
  target_link_libraries(${test_name} PRIVATE ${LD_FRAMEWORK_LIBRARIES})
  if(TARGET custom_layer)
    target_link_libraries(${test_name} PRIVATE custom_layer)
  endif()
  if(TARGET custom_loss)
    target_link_libraries(${test_name} PRIVATE custom_loss)
  endif()

  add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()
