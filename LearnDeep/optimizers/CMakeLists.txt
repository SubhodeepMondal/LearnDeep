add_library(optimizer SHARED optimizer_type.hpp optimizers.hpp optimizers.cpp sgd.cpp)

target_include_directories(optimizer PUBLIC ${CMAKE_SOURCE_DIR}/LearnDeep/)
target_link_directories(optimizer PUBLIC core absl::log absl::log_initialize absl::flags_parse)