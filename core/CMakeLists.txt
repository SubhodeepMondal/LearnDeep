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

add_library(core SHARED ${COMBINED_RESORCES})
target_link_libraries(core PUBLIC OpenMP::OpenMP_CXX)
target_include_directories(core PUBLIC ${CMAKE_SOURCE_DIR}/core)

enable_language(CUDA)
if(ENABLE_CUDA AND CMAKE_CUDA_COMPILER)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_FLAGS_DEBUG "-G -g -O0")
    set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG")
    message(STATUS "CUDA found: enabling CUDA support")
    enable_language(CUDA)
    set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
    add_compile_definitions(-DCUDA_ENABLED)

    add_library(gpu SHARED
        LAS/gpu_interface.cuh
        LAS/gpu_interface.cu
        LAS/gpu_micro_kernels.cuh
        LAS/gpu_micro_kernels.cu
    )

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


    # Link core with gpu (so test only needs to link core)
    target_link_libraries(core PRIVATE gpu)

    target_compile_definitions(core PUBLIC CUDA_ENABLED)

else()
    message(WARNING "CUDA not found: building CPU-only fallback")
endif()

