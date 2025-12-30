#include "cat.h"
#include <cassert>
#include <cstdio>
#include <hip/amd_detail/amd_hip_vector_types.h>

#define DIM 5000

bool test_catrix_add() {
  SField A = cat_alloc_host(DIM, DIM);
  SField dA = cat_adapt(A);
  SField dB = cat_adapt(A);
  SField dC = cat_adapt(A);

  cat_fill(dA, 1);
  cat_fill(dB, 2);

  cat_add(dC, dA, dB);
  cat_to_host(A, dC);

  bool result = true;
  for (int i = 0; i < DIM; i++)
    result &= CAT_AT(A, i, i).x == 5;

  cat_free_host(A);
  cat_free_device(dA);
  cat_free_device(dB);
  cat_free_device(dC);
  return true;
}

bool test_catrix_scale() {
  SField A = cat_alloc_host(DIM, DIM);
  SField dA = cat_adapt(A);
  SField dC = cat_adapt(A);

  cat_fill(dA, 1);

  cat_scale(dC, dA, 2);
  cat_to_host(A, dC);

  bool result = true;
  for (int i = 0; i < DIM; i++)
    result &= CAT_AT(A, i, i).x == 2;

  cat_free_host(A);
  cat_free_device(dA);
  cat_free_device(dC);
  return true;
}

bool test_catrix_lap() {
  SField A = cat_alloc_host(DIM, DIM);
  SField dA = cat_adapt(A);
  SField dC = cat_adapt(A);

  CAT_AT(A, 1, 1) = make_float2(1, 0);

  cat_to_device(dA, A);
  cat_lap(dC, dA);
  cat_to_host(A, dC);

  bool result = CAT_AT(A, 1, 2).x == 1;

  cat_free_host(A);
  cat_free_device(dA);
  cat_free_device(dC);
  return true;
}

bool test_catrix_naive() {
  SField A = cat_alloc_host(DIM, DIM);
  SField dA = cat_adapt(A);
  SField dB = cat_adapt(A);
  SField dC = cat_adapt(A);

  CAT_AT(A, 1, 1) = make_float2(1, 0);
  float2 idt = make_float2(0, 0.01);

  cat_to_device(dA, A);
  cat_lap(dB, dA);            // lap
  cat_scale(dC, dB, idt / 2); // idtlap/2
  cat_add(dB, dC, dA);
  cat_to_host(A, dC);

  bool result = CAT_AT(A, 1, 2).x == 0;

  cat_free_host(A);
  cat_free_device(dA);
  cat_free_device(dB);
  cat_free_device(dC);
  return true;
}

int main() {
  hipEvent_t start, stop;
  (void)hipEventCreate(&start);
  (void)hipEventCreate(&stop);

  struct {
    const char *name;
    bool (*fn)();
  } tests[] = {
      {"catrix_add", test_catrix_add},
      {"catrix_scale", test_catrix_scale},
      {"catrix_lap", test_catrix_lap},
      {"catrix_naive", test_catrix_naive},
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
