#include <chrono>
#include <iostream>

#include "hello.h"
#include "tensor.h"

int main(int argc, char **argv) {
  if (argc == 1)
    return 1;

  std::string mode = argv[1];
  if (mode == "1") {
    get_versions();
  } else if (mode == "2") {
    cuda_hello();
  } else if (mode == "3") {
    vector_add(TEST);
    cuda_vector_add(SINGLE_THREAD, TEST);
    cuda_vector_add(SINGLE_BLOCK, TEST);
    cuda_vector_add(MULTIPLE_BLOCKS, TEST);
  } else if (mode == "4") {
    std::chrono::time_point<std::chrono::high_resolution_clock> start, end;

    start = std::chrono::high_resolution_clock::now();
    vector_add();
    end = std::chrono::high_resolution_clock::now();
    std::cout << "vector_add took "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
              << " milliseconds" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    cuda_vector_add(SINGLE_THREAD);
    end = std::chrono::high_resolution_clock::now();
    std::cout << "cuda_vector_add SINGLE_THREAD took "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
              << " milliseconds" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    cuda_vector_add(SINGLE_BLOCK);
    end = std::chrono::high_resolution_clock::now();
    std::cout << "cuda_vector_add SINGLE_BLOCK took "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
              << " milliseconds" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    cuda_vector_add(MULTIPLE_BLOCKS);
    end = std::chrono::high_resolution_clock::now();
    std::cout << "cuda_vector_add MULTIPLE_BLOCKS took "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
              << " milliseconds" << std::endl;
  } else if (mode == "5") {
    std::chrono::time_point<std::chrono::high_resolution_clock> start, end;

    start = std::chrono::high_resolution_clock::now();
    vector_dot();
    end = std::chrono::high_resolution_clock::now();
    std::cout << "vector_dot took "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
              << " milliseconds" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    cuda_vector_dot();
    end = std::chrono::high_resolution_clock::now();
    std::cout << "cuda_vector_dot took "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
              << " milliseconds" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    cublas_vector_dot();
    end = std::chrono::high_resolution_clock::now();
    std::cout << "cublas_vector_dot took "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
              << " milliseconds" << std::endl;
  }

  get_errors();
  return 0;
}
