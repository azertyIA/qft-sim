#include <hip/hip_runtime.h>
#include <stddef.h>

#pragma once

typedef struct {
  size_t rows, cols, stride;
  float2 *data;
} SField;

typedef struct {
  SField x;
  SField y;
} VField;

typedef float2 (*SFieldFn)(float x, float y);

#define CAT_AT(m, i, j) (m).data[(i) * (m).stride + (j)]

SField s_alloc_host(size_t rows, size_t cols);
SField s_adapt(SField h);

void s_free_host(SField m);
void s_free_device(SField m);
void s_preview(SField m);

void s_to_device(SField d, SField h);
void s_to_host(SField h, SField d);
void s_host_fill(SField dst, SFieldFn f);
void s_clone(SField dst, const SField src);

void s_fill(SField m, float2 c);
void s_fill(SField m, float c);

void s_scale(SField dst, const SField a, float2 c);
void s_scale(SField dst, const SField a, float c);
void s_mult(SField dst, const SField a, const SField b);
void s_exp(SField dst, const SField a);

void s_norm(SField dst, const SField a, const float c);
void s_block(SField dst, const float2 v, const int2 x, const int2 y);

void s_add(SField dst, const SField a, const SField b);
void s_scale_add(SField dst, const SField a, const SField b, const float2 c);

void s_lap(SField dst, const SField u);
void s_filter(SField m);

void s_grad(VField dst, const SField a);
void v_div(SField dst, const VField a);
void v_dot(SField dst, const VField a, const VField b);
void v_dot_grad(SField dst, const VField a, const SField u);

void q_step_so(SField next, const SField curr, const SField prev,
               const SField h, const VField a, const float2 v2u_factor,
               const float2 h_factor);
void cg_step_so(const SField curr, const SField prev, const VField Tr,
                const VField Td, const SField O, const float d2_factor,
                const float h_factor);
