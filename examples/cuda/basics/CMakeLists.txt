cmake_minimum_required(VERSION 4.0)
project(cuda-basics LANGUAGES CXX CUDA)
set(EXECUTABLE_NAME cuda-basics)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CUDA_STANDARD 17)

add_library(libraries STATIC
    src/hello.cu
    src/tensor.cu
)
set_target_properties(libraries PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

add_executable(${EXECUTABLE_NAME} src/main.cpp)
target_link_libraries(${EXECUTABLE_NAME} PRIVATE libraries CUDA::cublas)
set_target_properties(${EXECUTABLE_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
