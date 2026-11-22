# CMakeLists.txt for the layer library
#
# Zero-edit onboarding: every *.cpp in this directory is compiled into the
# `layer` library. A layer registers itself through REGISTER_LAYER's static
# initializer, so adding a header + source here needs no CMake change.
#
# Out-of-tree layers are supported through two hooks:
#   -DEXTRA_LAYER_SOURCES="/path/to/a.cpp;/path/to/b.cpp"
#   -DEXTRA_LAYER_DIRS="/path/to/my_layers;/path/to/other_layers"
# Sources in EXTRA_LAYER_DIRS are globbed with CONFIGURE_DEPENDS, so dropping a
# new source into one of those directories is picked up on the next build.

file(GLOB LD_LAYER_SOURCES CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
file(GLOB LD_LAYER_HEADERS CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp")

set(LD_LAYER_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}")

foreach(layer_dir IN LISTS EXTRA_LAYER_DIRS)
  get_filename_component(layer_dir_abs "${layer_dir}" ABSOLUTE
                         BASE_DIR "${CMAKE_SOURCE_DIR}")
  file(GLOB LD_DIR_SOURCES CONFIGURE_DEPENDS "${layer_dir_abs}/*.cpp")
  file(GLOB LD_DIR_HEADERS CONFIGURE_DEPENDS "${layer_dir_abs}/*.hpp")
  list(APPEND LD_LAYER_SOURCES ${LD_DIR_SOURCES})
  list(APPEND LD_LAYER_HEADERS ${LD_DIR_HEADERS})
  list(APPEND LD_LAYER_INCLUDE_DIRS "${layer_dir_abs}")
endforeach()

foreach(layer_source IN LISTS EXTRA_LAYER_SOURCES)
  get_filename_component(layer_source_abs "${layer_source}" ABSOLUTE
                         BASE_DIR "${CMAKE_SOURCE_DIR}")
  list(APPEND LD_LAYER_SOURCES "${layer_source_abs}")
  get_filename_component(layer_source_dir "${layer_source_abs}" DIRECTORY)
  list(APPEND LD_LAYER_INCLUDE_DIRS "${layer_source_dir}")
endforeach()

list(REMOVE_DUPLICATES LD_LAYER_SOURCES)
list(REMOVE_DUPLICATES LD_LAYER_HEADERS)
list(REMOVE_DUPLICATES LD_LAYER_INCLUDE_DIRS)

add_library(layer SHARED ${LD_LAYER_SOURCES} ${LD_LAYER_HEADERS})
target_include_directories(layer PUBLIC ${CMAKE_SOURCE_DIR} ${LD_LAYER_INCLUDE_DIRS})
target_link_libraries(layer PUBLIC core)
