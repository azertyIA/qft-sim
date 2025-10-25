#include <hip/hip_runtime.h>
#include <stddef.h>

typedef struct {
  size_t rows, cols, stride;
  float2 *data;
} Catrix;

#define CAT_AT(m, i, j) (m).data[(i) * (m).stride + (j)]

Catrix cat_alloc_host(size_t rows, size_t cols);
Catrix cat_adapt(Catrix h);

void cat_free_host(Catrix m);
void cat_free_device(Catrix m);
void cat_preview(Catrix m);

void cat_to_device(Catrix d, Catrix h);
void cat_to_host(Catrix h, Catrix d);
void cat_clone(Catrix dst, const Catrix src);

void cat_fill(Catrix m, float2 c);
void cat_fill(Catrix m, float c);

void cat_scale(Catrix dst, const Catrix a, float2 c);
void cat_scale(Catrix dst, const Catrix a, float c);

void cat_norm(Catrix dst, const Catrix a, const float c);
void cat_block(Catrix dst, const float2 v, const int2 x, const int2 y);

void cat_add(Catrix dst, const Catrix a, const Catrix b);
void cat_lap(Catrix dst, const Catrix u);
void cat_step(Catrix next, const Catrix t, const Catrix u, const Catrix v,
              const float dt);
