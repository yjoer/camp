#ifndef CUDA_BASICS_TENSOR_H
#define CUDA_BASICS_TENSOR_H

enum VectorAddVariant {
  SINGLE_THREAD,
  SINGLE_BLOCK,
  MULTIPLE_BLOCKS,
};

enum VectorAddOutputMode {
  NONE,
  PRINT,
  TEST,
};

void vector_add(VectorAddOutputMode output_mode = NONE);
void cuda_vector_add(VectorAddVariant variant, VectorAddOutputMode output_mode = NONE);

#endif
