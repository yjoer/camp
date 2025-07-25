#include <cstdio>

#include "hello.h"

__global__ void _cuda_hello() {
  printf("Hello World from GPU!\n");
}

void cuda_hello() {
  _cuda_hello<<<1, 1>>>();
}

void get_versions() {
  int driver_version = 0;
  int runtime_version = 0;

  cudaDriverGetVersion(&driver_version);
  cudaRuntimeGetVersion(&runtime_version);

  printf("CUDA Driver Version: %d\n", driver_version);
  printf("CUDA Runtime Version: %d\n", runtime_version);
}

void get_errors() {
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    printf(cudaGetErrorString(err));
}
