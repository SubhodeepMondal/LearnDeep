
enable_language(CXX)

add_library(tensor_api SHARED
    tensor.h
    tensor.cpp
)

target_include_directories(tensor_api PUBLIC ${CMAKE_SOURCE_DIR}/core)


target_link_libraries(tensor_api
    PUBLIC
    core
    absl::log
    absl::log_initialize
    absl::flags_parse
)
