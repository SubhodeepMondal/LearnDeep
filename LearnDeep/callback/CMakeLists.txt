
# CMakeLists.txt for the model library

set(ATTACHED_FILES
    callback.hpp
    callbacktrace.cc
    callbackearlystopping.cc)

add_library(callback SHARED ${ATTACHED_FILES})
target_include_directories(callback PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(callback PUBLIC  core layer absl::log absl::log_initialize absl::flags_parse)