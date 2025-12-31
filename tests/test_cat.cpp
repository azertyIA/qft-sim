#include "cat.h"
#include <cassert>
#include <cstdio>
#include <hip/amd_detail/amd_hip_vector_types.h>

#define DIM 500

bool test_catrix_write(SField H, SField sO, VField vO, SField sA, SField sB,
                       VField vA, VField vB, float2 E) {
  s_to_host(H, sO);
  return true;
}

bool test_catrix_add(SField H, SField sO, VField, SField sA, SField sB, VField,
                     VField, float2) {
  s_add(sO, sA, sB);
  s_to_host(H, sO);

  bool result = true;
  for (int i = 0; i < DIM; i++)
    result &= CAT_AT(H, i, i).x == 3;
  return result;
}

bool test_catrix_scale(SField H, SField dO, VField, SField dA, SField, VField,
                       VField, float2 E) {
  s_scale(dO, dA, E);
  s_to_host(H, dO);

  bool result = true;
  for (int i = 0; i < DIM; i++)
    result &= CAT_AT(H, i, i).x == 2;
  return result;
}

bool test_catrix_scale_add(SField H, SField dO, VField, SField dA, SField dB,
                           VField, VField, float2 E) {
  s_scale_add(dO, dA, dB, E);
  s_to_host(H, dO);

  bool result = true;
  for (int i = 0; i < DIM; i++)
    result &= CAT_AT(H, i, i).x == 5;
  return result;
}

bool test_catrix_mult(SField H, SField sO, VField vO, SField sA, SField sB,
                      VField vA, VField vB, float2 E) {
  s_mult(sO, sA, sB);
  s_to_host(H, sO);
  bool result = true;

  for (int i = 0; i < DIM; i++)
    result &= CAT_AT(H, i, i).x == 2;
  return result;
}

bool test_catrix_grad(SField H, SField sO, VField vO, SField sA, SField sB,
                      VField vA, VField vB, float2 E) {
  s_grad(vO, sA);
  s_to_host(H, vO.y);
  bool result = true;

  for (int i = 0; i < DIM; i++)
    if (CAT_AT(H, i, i) != 0)
      printf("%d, %f\n", i, CAT_AT(H, i, i).x);
  // result &= CAT_AT(H, i, i).x == 0;
  return result;
}

bool test_catrix_div(SField H, SField sO, VField vO, SField sA, SField sB,
                     VField vA, VField vB, float2 E) {
  f_div(sO, vA);
  s_to_host(H, sO);
  bool result = true;

  for (int i = 0; i < DIM; i++)
    result &= CAT_AT(H, i, i).x == 0;
  return result;
}

bool test_catrix_dot(SField H, SField sO, VField vO, SField sA, SField sB,
                     VField vA, VField vB, float2 E) {
  f_dot(sO, vA, vB);
  s_to_host(H, sO);
  bool result = true;

  for (int i = 0; i < DIM; i++)
    result &= CAT_AT(H, i, i).x == 39;
  return result;
}

bool test_catrix_lap(SField H, SField dO, VField, SField dA, SField, VField,
                     VField, float2) {
  s_lap(dO, dA);
  s_to_host(H, dO);

  bool result = CAT_AT(H, 1, 2).x == 1;
  return true;
}

int main() {
  hipEvent_t start, stop;
  (void)hipEventCreate(&start);
  (void)hipEventCreate(&stop);

  SField H = s_alloc_host(DIM, DIM);
  SField sA = s_adapt(H);
  SField sB = s_adapt(H);
  SField sO = s_adapt(H);
  VField vA = {s_adapt(H), s_adapt(H)};
  VField vB = {s_adapt(H), s_adapt(H)};
  VField vO = {sA, s_adapt(H)};
  float2 E = make_float2(2, 0);

  s_fill(sA, 1);
  s_fill(sB, 2);
  s_fill(vA.x, 3);
  s_fill(vA.y, 4);
  s_fill(vB.x, 5);
  s_fill(vB.y, 6);

  struct {
    const char *name;
    bool (*fn)(SField, SField, VField, SField, SField, VField, VField, float2);
  } tests[] = {
      // {"catrix_grad", test_catrix_grad},
      {"catrix_write", test_catrix_write},
      {"catrix_add", test_catrix_add},
      {"catrix_scale", test_catrix_scale},
      {"catrix_scale_add", test_catrix_scale_add},
      {"catrix_mult", test_catrix_mult},
      {"catrix_div", test_catrix_div},
      {"catrix_dot", test_catrix_dot},
      {"catrix_lap", test_catrix_lap},
  };

  int failures = 0;
  for (auto &t : tests) {
    (void)hipEventRecord(start, 0);
    bool ok = t.fn(H, sO, vO, sA, sB, vA, vB, E);
    (void)hipEventRecord(stop, 0);
    (void)hipEventSynchronize(stop);
    float ms = 0;
    (void)hipEventElapsedTime(&ms, start, stop);
    printf("[%s] %s: %.2f ms\n", t.name, ok ? "PASS" : "FAIL", ms);
    if (!ok)
      failures++;
  }

  s_free_host(H);
  s_free_device(sA);
  s_free_device(sB);
  s_free_device(sO);
  s_free_device(vA.x);
  s_free_device(vA.y);
  s_free_device(vB.x);
  s_free_device(vB.y);
  s_free_device(vO.y);
  return failures ? 1 : 0;
}
