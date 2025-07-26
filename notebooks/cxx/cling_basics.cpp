// %% [markdown]
// ## Cling Options

// %%
#include "cling/Interpreter/Interpreter.h"

// %%
gCling->getDefaultOptLevel();

// %%
gCling->setDefaultOptLevel(0);

// %%
gClingOpts->AllowRedefinition = 1;

// %%
gClingOpts->AllowRedefinition;

// %% [markdown]
// ## Hello, World!

// %%
#include <iostream>

// %%
std::cout << "Hello, World!" << std::endl;

// %% [markdown]
// ## Global and Local Variables

// %%
int p = 1;
std::cout << p << std::endl;

{
  int p = 2;
  std::cout << p << std::endl;
}

std::cout << p << std::endl;

// %% [markdown]
// ## Standard C++ Features

// %%
int add(int a, int b) {
  return a + b;
}

// %%
std::cout << add(1, 2);

// %%
class Rectangle {
  int width;
  int height;

public:
  Rectangle(int w, int h) : width(w), height(h) {}

  int area() {
    return width * height;
  }
}

// %%
Rectangle rect(2, 4);
std::cout << rect.area();

// %% [markdown]
// ## Persistent Memory

// %%
int k = 0;

// %%
for (int end = k + 5; k < end; k++) {
  std::cout << k << std::endl;
}

// %% [markdown]
// ## Templates

// %%
#include <sstream>

// %%
template <class T> class A {};

template <class T> std::string print_value(A<T> *) {
  std::ostringstream oss;
  oss << "Hello, World!";
  return oss.str();
}

// %%
A<int> a;
A<int> *a_ptr = &a;
print_value(a_ptr);

// %% [markdown]
// ## Template Specialization

// %%
#include <chrono>

// %%
constexpr int dim = 512;
float *a = new float[dim * dim];
float *b = new float[dim * dim];
float *c = new float[dim * dim];

// %%
for (int i = 0; i < dim; ++i) {
  a[i] = static_cast<float>(i + 1);
  b[i] = static_cast<float>(i + 1);
}

// %%
void matmul(float const *const a, float const *const b, float *const c, const int dim) {
  float sum = 0.f;
  for (int i = 0; i < dim; ++i) {
    for (int j = 0; j < dim; ++j) {
      for (int k = 0; k < dim; ++k) {
        sum += a[i * dim + k] * b[k * dim + j];
      }
      c[i * dim + j] = sum;
      sum = 0.f;
    }
  }
}

// %%
template <int dim>
void matmul_t(float const *const a, float const *const b, float *const c) {
  float sum = 0.f;
  for (int i = 0; i < dim; ++i) {
    for (int j = 0; j < dim; ++j) {
      for (int k = 0; k < dim; ++k) {
        sum += a[i * dim + k] * b[k * dim + j];
      }
      c[i * dim + j] = sum;
      sum = 0.f;
    }
  }
}

// %%
std::chrono::time_point<std::chrono::high_resolution_clock> v_start, v_end;
std::chrono::time_point<std::chrono::high_resolution_clock> t_start, t_end;

// %%
v_start = std::chrono::high_resolution_clock::now();
matmul(a, b, c, dim);
v_end = std::chrono::high_resolution_clock::now();

t_start = std::chrono::high_resolution_clock::now();
matmul_t<dim>(a, b, c);
t_end = std::chrono::high_resolution_clock::now();

// %%
std::chrono::duration<double> v_diff = v_end - v_start;
std::chrono::duration<double> t_diff = t_end - t_start;

std::cout << "v_matmul: " << v_diff.count() << "s" << std::endl
          << "t_matmul: " << t_diff.count() << "s" << std::endl;

// %% [markdown]
// ## Reflection

// %%
struct Point {
  int x;
  int y;
};

Point q;
q.x = 1;
q.y = 2;
q;

// %% [markdown]
// ## Input Request

// %%
#include <string>

// %%
std::string s;
std::cin >> s;

// %%
std::cout << s;

// %%
