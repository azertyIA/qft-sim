#include "mat.h"
#include <cassert>
#include <cstdio>
#include <hip/amd_detail/amd_hip_runtime.h>
#include <hip/hip_runtime.h>

#define DIM 500

bool test_matrix_fill() {
  Matrix A = mat_alloc_host(DIM, DIM);
  Matrix dA = mat_adapt(A);

  mat_fill_device(dA, 2);
  mat_copy_to_host(A, dA);

  bool result = true;
  for (int i = 0; i < DIM; i++)
    result &= MAT_AT(A, i, i) == 2;

  mat_free_host(A);
  mat_free_device(dA);
  return result;
}

bool test_matrix_add() {
  Matrix A = mat_alloc_host(DIM, DIM);
  Matrix dA = mat_adapt(A);
  Matrix dB = mat_adapt(A);
  Matrix dC = mat_adapt(A);

  mat_fill_device(dA, 2);
  mat_fill_device(dB, 3);

  mat_add_device(dC, dA, dB);
  mat_copy_to_host(A, dC);

  bool result = true;
  for (int i = 0; i < DIM; i++)
    result &= MAT_AT(A, i, i) == 5;

  mat_free_host(A);
  mat_free_device(dA);
  mat_free_device(dB);
  mat_free_device(dC);
  return result;
}

bool test_matrix_mult() {
  Matrix A = mat_alloc_host(DIM, DIM);
  Matrix dA = mat_adapt(A);
  Matrix dB = mat_adapt(A);
  Matrix dC = mat_adapt(A);

  mat_fill_device(dA, 2);
  mat_fill_device(dB, 4);

  mat_mult_device(dC, dA, dB);
  mat_copy_to_host(A, dC);
  bool result = MAT_AT(A, 0, 0) == 8 * DIM;

  mat_free_host(A);
  mat_free_device(dA);
  mat_free_device(dB);
  mat_free_device(dC);
  return result;
}

int main() {
  hipEvent_t start, stop;
  (void)hipEventCreate(&start);
  (void)hipEventCreate(&stop);

  struct {
    const char *name;
    bool (*fn)();
  } tests[] = {
      {"matrix_fill", test_matrix_fill},
      {"matrix_add", test_matrix_add},
      {"matrix_mult", test_matrix_mult},
  };

  int failures = 0;
  for (auto &t : tests) {
    (void)hipEventRecord(start, 0);
    bool ok = t.fn();
    (void)hipEventRecord(stop, 0);
    (void)hipEventSynchronize(stop);
    float ms = 0;
    (void)hipEventElapsedTime(&ms, start, stop);
    printf("[%s] %s: %.2f ms\n", t.name, ok ? "PASS" : "FAIL", ms);
    if (!ok)
      failures++;
  }
  return failures ? 1 : 0;
}
