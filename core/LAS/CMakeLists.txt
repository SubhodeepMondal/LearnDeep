cmake_minimum_required(VERSION 3.15)
project(las)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(OpenMP REQUIRED)  # Find OpenMP package

add_library(${PROJECT_NAME} CPULibrary.cpp)
if(OpenMP_CXX_FOUND)
    target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX)  # Link OpenMP
    target_compile_options(${PROJECT_NAME} PRIVATE ${OpenMP_CXX_FLAGS})
endif()

set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/../../)
install(TARGETS ${PROJECT_NAME}
        LIBRARY DESTINATION lib)