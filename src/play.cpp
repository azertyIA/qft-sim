#include "quantum.h"
#include <cstddef>
#include <cstdio>
#include <hip/amd_detail/amd_hip_vector_types.h>
#include <hip/hip_runtime.h>

int main(int argc, char *argv[]) {
  size_t dim = 50;
  SField h = s_alloc_host(dim, dim);
  VField VU = {s_adapt(h), s_adapt(h)};
  VField A = {s_adapt(h), s_adapt(h)};
  SField U = s_adapt(h);
  SField O = s_adapt(h);
  float2 e = make_float2(1, 2);
  hipEvent_t start, stop;
  float ms = 0;

  (void)hipEventRecord(start, 0);
  s_scale_add(O, A.x, U, e);
  (void)hipEventRecord(stop, 0);
  (void)hipEventSynchronize(stop);
  (void)hipEventElapsedTime(&ms, start, stop);
  printf("scale_add: %.6f ms\n", ms);

  (void)hipEventRecord(start, 0);
  s_scale(U, U, e);
  s_add(O, U, A.x);
  (void)hipEventRecord(stop, 0);
  (void)hipEventSynchronize(stop);
  (void)hipEventElapsedTime(&ms, start, stop);
  printf("scale.add: %.6f ms\n", ms);

  return 0;
}
