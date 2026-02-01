# loss library

add_library(loss SHARED loss.hpp squared_error.cpp)
target_include_directories(loss PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(loss PUBLIC core absl::log absl::log_initialize absl::flags_parse)