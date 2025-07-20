cmake_minimum_required(VERSION 3.21)
project(deeplearningLA CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2")
find_package(OpenMP REQUIRED) # Find OpenMP package

add_subdirectory(thirdparty/googletest)


add_subdirectory(core)

add_subdirectory(api)

add_subdirectory(test)

add_custom_target(
    src
    DEPENDS kernel graph framework las gpu
)

add_custom_target(
    test
    DEPENDS kernel graph framework las gpu tensor_api
)

set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/build/)
install(TARGETS kernel
    graph
    framework
    las
    gpu
    tensor_api
    LIBRARY DESTINATION lib)
