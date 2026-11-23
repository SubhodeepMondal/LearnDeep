# CMakeLists.txt for the optimizer library
#
# Zero-edit onboarding: every *.cpp in this directory is compiled into the
# `optimizer` library. An optimizer registers itself through
# REGISTER_INTERNAL_OPTIMIZER's static initializer, so adding a header + source
# here needs no CMake change.
#
# Out-of-tree optimizers are reserved for a later phase and will use hooks
# analogous to EXTRA_LOSS_SOURCES / EXTRA_LOSS_DIRS.

file(GLOB LD_OPTIMIZER_SOURCES CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
file(GLOB LD_OPTIMIZER_HEADERS CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp")

add_library(optimizer SHARED ${LD_OPTIMIZER_SOURCES} ${LD_OPTIMIZER_HEADERS})

target_include_directories(optimizer PUBLIC ${CMAKE_SOURCE_DIR}/LearnDeep/)
target_link_libraries(optimizer PUBLIC core absl::log absl::log_initialize
                                             absl::flags_parse)
