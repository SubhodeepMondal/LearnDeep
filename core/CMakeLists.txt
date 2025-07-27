# CMakeLists.txt for the core library

#creating library for kernel operations

set(COMBINED_RESORCES
    kernel/opskernel.h
    kernel/opskernel.cpp
    kernel/add.cpp
    kernel/matmul.cpp
    kernel/mul.cpp
    kernel/power.cpp
    kernel/reducesum.cpp
    kernel/scale.cpp


    graph/graph_framework.hpp
    graph/graph_framework.cpp
    graph/graph_node.hpp
    graph/graph_node.cpp


    framework/MathLibrary.h
    framework/MathLibrary.cpp
    framework/NDynamicArray.h
    framework/NDynamicArray.cpp


    LAS/CPULibrary.h
    LAS/CPULibrary.cpp
    LAS/avx2_micro_kernels.h
    LAS/avx2_micro_kernels.cpp
)
# add_library(kernel SHARED
#     kernel/opskernel.h
#     kernel/opskernel.cpp
#     kernel/add.cpp
#     kernel/matmul.cpp
#     kernel/mul.cpp
#     kernel/power.cpp
#     kernel/reducesum.cpp
#     kernel/scale.cpp
# )

# #creating library for graphs
# add_library(graph SHARED
#     graph/graph_framework.hpp
#     graph/graph_framework.cpp
#     graph/graph_node.hpp
#     graph/graph_node.cpp
# )

# #creating library for framework operations
# add_library(framework SHARED
#     framework/MathLibrary.h
#     framework/MathLibrary.cpp
#     framework/NDynamicArray.h
#     framework/NDynamicArray.cpp
# )

# # creating library Linear Algebra Subroutines(LAS)
# add_library(las SHARED
#     LAS/CPULibrary.h
#     LAS/CPULibrary.cpp
#     LAS/avx2_micro_kernels.h
#     LAS/avx2_micro_kernels.cpp
# )

find_package(CUDA QUIET)
if(CUDA_FOUND)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    message(STATUS "CUDA found: enabling CUDA support")
    enable_language(CUDA)
    set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
    add_compile_definitions(-DCUDA_ENABLED)

    list(APPEND COMBINED_RESORCES
        LAS/gpu_interface.cuh
        LAS/gpu_interface.cu
        LAS/gpu_micro_kernels.cuh
        LAS/gpu_micro_kernels.cu
    )
    add_library(gpu SHARED LAS/gpu_interface.cu LAS/gpu_micro_kernels.cu)

    # Enable separate compilation for nested kernel launches
    set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

    set_target_properties(gpu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

    # Suppress deprecated arch warning
    target_compile_options(gpu PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:--relocatable-device-code=true>")
    add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-gpu-targets>")

    target_include_directories(gpu PRIVATE ${CMAKE_SOURCE_DIR}/core/
        /usr/local/cuda/include)

    # Or set architecture explicitly
    set(CMAKE_CUDA_ARCHITECTURES 90) # or 80, 86, etc.

else()
    message(WARNING "CUDA not found: building CPU-only fallback")
endif()

add_library(core SHARED ${COMBINED_RESORCES})
target_include_directories(core PUBLIC ${CMAKE_SOURCE_DIR}/core)




# target_include_directories(framework PUBLIC ${CMAKE_SOURCE_DIR}/core/)
# target_include_directories(graph PUBLIC ${CMAKE_SOURCE_DIR}/core/)
# target_include_directories(kernel PUBLIC ${CMAKE_SOURCE_DIR}/core/)
