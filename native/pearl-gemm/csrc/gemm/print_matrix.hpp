#pragma once
#include <cstdio>

#include <cute/tensor.hpp>

template <typename T, int PAD = 4>
CUTE_DEVICE void printMatrixDevice(const T* matrix_ptr, int num_rows,
                                   int num_cols) {
  printf("       ");
  for (int col = 0; col < num_cols; ++col) {
    printf("%*d ", PAD, col);
  }
  printf("\n");

  printf("       ");
  for (int col = 0; col < num_cols; ++col) {
    for (int i = 0; i < PAD; ++i) {
      printf("-");
    }
  }
  printf("\n");

  for (int row = 0; row < num_rows; ++row) {
    printf("%4d | ", row);
    for (int col = 0; col < num_cols; ++col) {
      printf("%*d ", PAD, static_cast<int>(matrix_ptr[row * num_cols + col]));
    }
    printf("\n");
  }
}

__device__ void printTensorDevice(const auto& tensor) {
  for (int i = 0; i < size<0>(tensor); ++i) {
    for (int j = 0; j < size<1>(tensor); ++j) {
      printf("%d ", static_cast<int>(tensor(i, j)));
    }
    printf("\n");
  }
}
