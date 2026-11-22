# CMakeLists.txt for the loss library
#
# Zero-edit onboarding: every *.cpp in this directory is compiled into the
# `loss` library. A loss registers itself through REGISTER_INTERNAL_LOSS's
# static initializer, so adding a header + source here needs no CMake change.
#
# Out-of-tree losses are supported through two hooks:
#   -DEXTRA_LOSS_SOURCES="/path/to/a.cpp;/path/to/b.cpp"
#   -DEXTRA_LOSS_DIRS="/path/to/my_losses;/path/to/other_losses"

file(GLOB LD_LOSS_SOURCES CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
file(GLOB LD_LOSS_HEADERS CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp")

set(LD_LOSS_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}")

foreach(loss_dir IN LISTS EXTRA_LOSS_DIRS)
  get_filename_component(loss_dir_abs "${loss_dir}" ABSOLUTE
                         BASE_DIR "${CMAKE_SOURCE_DIR}")
  file(GLOB LD_DIR_SOURCES CONFIGURE_DEPENDS "${loss_dir_abs}/*.cpp")
  file(GLOB LD_DIR_HEADERS CONFIGURE_DEPENDS "${loss_dir_abs}/*.hpp")
  list(APPEND LD_LOSS_SOURCES ${LD_DIR_SOURCES})
  list(APPEND LD_LOSS_HEADERS ${LD_DIR_HEADERS})
  list(APPEND LD_LOSS_INCLUDE_DIRS "${loss_dir_abs}")
endforeach()

foreach(loss_source IN LISTS EXTRA_LOSS_SOURCES)
  get_filename_component(loss_source_abs "${loss_source}" ABSOLUTE
                         BASE_DIR "${CMAKE_SOURCE_DIR}")
  list(APPEND LD_LOSS_SOURCES "${loss_source_abs}")
  get_filename_component(loss_source_dir "${loss_source_abs}" DIRECTORY)
  list(APPEND LD_LOSS_INCLUDE_DIRS "${loss_source_dir}")
endforeach()

list(REMOVE_DUPLICATES LD_LOSS_SOURCES)
list(REMOVE_DUPLICATES LD_LOSS_HEADERS)
list(REMOVE_DUPLICATES LD_LOSS_INCLUDE_DIRS)

add_library(loss SHARED ${LD_LOSS_SOURCES} ${LD_LOSS_HEADERS})
target_include_directories(loss PUBLIC ${CMAKE_SOURCE_DIR} ${LD_LOSS_INCLUDE_DIRS})
target_link_libraries(loss PUBLIC core absl::log absl::log_initialize absl::flags_parse)
