
# CMakeLists.txt for the model library

add_library(callback SHARED callback.hpp callback.cpp)
target_include_directories(callback PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(callback PUBLIC  core layer absl::log absl::log_initialize absl::flags_parse)