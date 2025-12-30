#include <hip/hip_runtime.h>
#include <stddef.h>

typedef struct {
  size_t rows, cols, stride;
  float2 *data;
} SField;

typedef struct {
  SField x;
  SField y;
} VField;

typedef float2 (*CatFn)(float x, float y);

#define CAT_AT(m, i, j) (m).data[(i) * (m).stride + (j)]

SField cat_alloc_host(size_t rows, size_t cols);
SField cat_adapt(SField h);

void cat_free_host(SField m);
void cat_free_device(SField m);
void cat_preview(SField m);

void cat_to_device(SField d, SField h);
void cat_to_host(SField h, SField d);
void cat_host_fill(SField dst, CatFn f);
void cat_clone(SField dst, const SField src);

void cat_fill(SField m, float2 c);
void cat_fill(SField m, float c);

void cat_scale(SField dst, const SField a, float2 c);
void cat_scale(SField dst, const SField a, float c);
void cat_mult(SField dst, const SField a, const SField b);

void cat_norm(SField dst, const SField a, const float c);
void cat_block(SField dst, const float2 v, const int2 x, const int2 y);

void cat_add(SField dst, const SField a, const SField b);
void cat_augment(SField dst, const SField a, const SField b, const float2 c);
void cat_hdot(SField dst, const SField a, const SField b);

void cat_lap(SField dst, const SField u);
void cat_step(SField next, const SField t, const SField u, const SField v,
              const float dt);
void cat_filter(SField m);

void cat_grad(VField dst, const SField a);
void cfd_div(SField dst, const VField a);
void cfd_dot(SField dst, const VField a, const VField b);
void cfd_dot_grad(SField dst, const VField a, const SField b);
