# CMakeLists.txt for the model library

add_library(model SHARED model.hpp model.cpp)
target_include_directories(model PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(model PUBLIC  core layer absl::log absl::log_initialize absl::flags_parse)