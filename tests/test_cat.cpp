#include "cat.h"
#include <cassert>
#include <cstdio>
#include <hip/amd_detail/amd_hip_vector_types.h>

#define DIM 5000

bool test_catrix_add() {
  SField A = s_alloc_host(DIM, DIM);
  SField dA = s_adapt(A);
  SField dB = s_adapt(A);
  SField dC = s_adapt(A);

  s_fill(dA, 1);
  s_fill(dB, 2);

  s_add(dC, dA, dB);
  s_to_host(A, dC);

  bool result = true;
  for (int i = 0; i < DIM; i++)
    result &= CAT_AT(A, i, i).x == 5;

  s_free_host(A);
  s_free_device(dA);
  s_free_device(dB);
  s_free_device(dC);
  return true;
}

bool test_catrix_scale() {
  SField A = s_alloc_host(DIM, DIM);
  SField dA = s_adapt(A);
  SField dC = s_adapt(A);

  s_fill(dA, 1);

  s_scale(dC, dA, 2);
  s_to_host(A, dC);

  bool result = true;
  for (int i = 0; i < DIM; i++)
    result &= CAT_AT(A, i, i).x == 2;

  s_free_host(A);
  s_free_device(dA);
  s_free_device(dC);
  return true;
}

bool test_catrix_lap() {
  SField A = s_alloc_host(DIM, DIM);
  SField dA = s_adapt(A);
  SField dC = s_adapt(A);

  CAT_AT(A, 1, 1) = make_float2(1, 0);

  s_to_device(dA, A);
  s_lap(dC, dA);
  s_to_host(A, dC);

  bool result = CAT_AT(A, 1, 2).x == 1;

  s_free_host(A);
  s_free_device(dA);
  s_free_device(dC);
  return true;
}

bool test_catrix_naive() {
  SField A = s_alloc_host(DIM, DIM);
  SField dA = s_adapt(A);
  SField dB = s_adapt(A);
  SField dC = s_adapt(A);

  CAT_AT(A, 1, 1) = make_float2(1, 0);
  float2 idt = make_float2(0, 0.01);

  s_to_device(dA, A);
  s_lap(dB, dA);            // lap
  s_scale(dC, dB, idt / 2); // idtlap/2
  s_add(dB, dC, dA);
  s_to_host(A, dC);

  bool result = CAT_AT(A, 1, 2).x == 0;

  s_free_host(A);
  s_free_device(dA);
  s_free_device(dB);
  s_free_device(dC);
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
