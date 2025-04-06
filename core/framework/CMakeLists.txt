cmake_minimum_required(VERSION 3.15)
project(framework)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/../../)
add_library(${PROJECT_NAME} STATIC 
                MathLibrary.cpp
                NDynamicArray.cpp)

install(TARGETS ${PROJECT_NAME}  
        LIBRARY DESTINATION lib)