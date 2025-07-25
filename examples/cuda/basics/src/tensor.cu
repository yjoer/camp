#include <cassert>
#include <cstdio>

#include "tensor.h"

void _vector_add(int n, float *a, float *b, float *y) {
  for (int i = 0; i < n; i++) {
    y[i] = a[i] + b[i];
  }
}

void vector_add(VectorAddOutputMode output_mode) {
  int n = 10'000'000;

  float *a, *b, *y;
  a = (float *)malloc(sizeof(float) * n);
  b = (float *)malloc(sizeof(float) * n);
  y = (float *)malloc(sizeof(float) * n);

  for (int i = 0; i < n; i++) {
    a[i] = 1.f;
    b[i] = 2.f;
  }

  _vector_add(n, a, b, y);

  if (output_mode == PRINT) {
    printf("%f, %f, %f\n", y[0], y[n / 2], y[n - 1]);
  } else if (output_mode == TEST) {
    for (int i = 0; i < n; i++) {
      assert(fabs(y[i] - a[i] - b[i]) < 1e-6);
    }

    printf("All values are correct.\n");
  }

  free(a);
  free(b);
  free(y);
}

__global__ void _cuda_vector_add(int n, float *a, float *b, float *y) {
  for (int i = 0; i < n; i++) {
    y[i] = a[i] + b[i];
  }
}

__global__ void _cuda_vector_add_sb(int n, float *a, float *b, float *y) {
  // 0, 256, 512, 768, 1024, ...
  int index = threadIdx.x;
  int stride = blockDim.x;

  for (int i = index; i < n; i += stride) {
    y[i] = a[i] + b[i];
  }
}

__global__ void _cuda_vector_add_mb(int n, float *a, float *b, float *y) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = index; i < n; i += stride) {
    y[i] = a[i] + b[i];
  }
}

void cuda_vector_add(VectorAddVariant variant, VectorAddOutputMode output_mode) {
  int n = 10'000'000;

  float *a, *b, *y;
  a = (float *)malloc(sizeof(float) * n);
  b = (float *)malloc(sizeof(float) * n);
  y = (float *)malloc(sizeof(float) * n);

  for (int i = 0; i < n; i++) {
    a[i] = 1.f;
    b[i] = 2.f;
  }

  float *device_a, *device_b, *device_y;
  cudaMalloc((void **)&device_a, sizeof(float) * n);
  cudaMemcpy(device_a, a, sizeof(float) * n, cudaMemcpyHostToDevice);

  cudaMalloc((void **)&device_b, sizeof(float) * n);
  cudaMemcpy(device_b, b, sizeof(float) * n, cudaMemcpyHostToDevice);

  cudaMalloc((void **)&device_y, sizeof(float) * n);

  if (variant == SINGLE_THREAD) {
    _cuda_vector_add<<<1, 1>>>(n, device_a, device_b, device_y);
  } else if (variant == SINGLE_BLOCK) {
    _cuda_vector_add_sb<<<1, 256>>>(n, device_a, device_b, device_y);
  } else if (variant == MULTIPLE_BLOCKS) {
    int n_blocks = (n + 256 - 1) / 256;
    _cuda_vector_add_mb<<<n_blocks, 256>>>(n, device_a, device_b, device_y);
  }

  cudaDeviceSynchronize();

  if (output_mode == PRINT) {
    float out_a, out_b, out_c;
    cudaMemcpy(&out_a, device_y, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&out_b, device_y + (n / 2), sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&out_c, device_y + (n - 1), sizeof(float), cudaMemcpyDeviceToHost);

    printf("%f, %f, %f\n", out_a, out_b, out_c);
  } else if (output_mode == TEST) {
    cudaMemcpy(y, device_y, sizeof(float) * n, cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; i++) {
      assert(fabs(y[i] - a[i] - b[i]) < 1e-6);
    }

    printf("All values are correct.\n");
  }

  cudaFree(device_a);
  free(a);

  cudaFree(device_b);
  free(b);

  cudaFree(device_y);
  free(y);
}
