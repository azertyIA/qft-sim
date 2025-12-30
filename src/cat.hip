#include "cat.h"
#include <__clang_hip_runtime_wrapper.h>
#include <assert.h>
#include <cstddef>
#include <cstdio>
#include <hip/amd_detail/amd_hip_vector_types.h>
#include <hip/amd_detail/host_defines.h>
#include <stdlib.h>

#define HIP_CHECK(command)                                                     \
  {                                                                            \
    hipError_t status = command;                                               \
    assert(status == hipSuccess);                                              \
  }

__device__ __host__ inline float2 cmul(float2 a, float2 b) {
  return make_float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

SField cat_alloc_host(size_t rows, size_t cols) {
  SField m;
  m.rows = rows;
  m.cols = cols;
  m.stride = cols;
  m.data = (float2 *)malloc(rows * cols * sizeof(float2));
  return m;
}

SField cat_adapt(SField h) {
  SField m;
  m.rows = h.rows;
  m.cols = h.cols;
  m.stride = h.cols;
  HIP_CHECK(hipMalloc(&m.data, h.rows * h.cols * sizeof(float2)));
  return m;
}

void cat_free_host(SField m) { free(m.data); }

void cat_free_device(SField m) { HIP_CHECK(hipFree(m.data)); }

void cat_preview(SField m) {
  for (int r = 0; r < 4 && r < m.rows; ++r) {
    for (int c = 0; c < 4 && c < m.cols; ++c) {
      float2 v = CAT_AT(m, r, c);
      printf("(%.2f + %.2fi)\t", v.x, v.y);
    }
    printf("\n");
  }
  printf("\n");
}

void cat_to_device(SField d, SField h) {
  size_t total = d.rows * d.cols;
  HIP_CHECK(
      hipMemcpy(d.data, h.data, total * sizeof(float2), hipMemcpyHostToDevice));
}

void cat_to_host(SField h, SField d) {
  size_t total = d.rows * d.cols;
  HIP_CHECK(
      hipMemcpy(h.data, d.data, total * sizeof(float2), hipMemcpyDeviceToHost));
}

void cat_host_fill(SField dst, CatFn f) {
  for (size_t row = 0; row < dst.rows; row++) {
    for (size_t col = 0; col < dst.cols; col++) {
      CAT_AT(dst, row, col) = f(col, row);
    }
  }
}

void cat_clone(SField dst, const SField src) {
  assert(dst.rows == src.rows && dst.cols == src.cols);
  HIP_CHECK(hipMemcpy(dst.data, src.data, dst.rows * dst.cols * sizeof(float2),
                      hipMemcpyDeviceToDevice));
}

__global__ void cat_fill_kernel(SField m, float2 c) {
  size_t row = blockIdx.y * blockDim.y + threadIdx.y;
  size_t col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= m.rows || col >= m.cols)
    return;
  CAT_AT(m, row, col) = c;
}

void cat_fill(SField m, float2 c) {
  dim3 threads(32, 32);
  dim3 blocks((m.cols + threads.x - 1) / threads.x,
              (m.rows + threads.y - 1) / threads.y);
  hipLaunchKernelGGL(cat_fill_kernel, blocks, threads, 0, 0, m, c);
}

void cat_fill(SField m, float c) { cat_fill(m, make_float2(c, 0)); }

__global__ void cat_block_kernel(SField m, const float2 v, const int2 x,
                                 const int2 y) {
  size_t row = blockIdx.y * blockDim.y + threadIdx.y;
  size_t col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= y.y || row <= y.x || col >= x.y || col <= x.x)
    return;
  CAT_AT(m, row, col) += v;
}

void cat_block(SField m, const float2 v, const int2 x, const int2 y) {
  dim3 threads(32, 32);
  dim3 blocks((m.cols + threads.x - 1) / threads.x,
              (m.rows + threads.y - 1) / threads.y);
  hipLaunchKernelGGL(cat_block_kernel, blocks, threads, 0, 0, m, v, x, y);
}

__global__ void cat_add_kernel(SField dst, SField a, SField b) {
  size_t row = blockIdx.y * blockDim.y + threadIdx.y;
  size_t col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= a.rows || col >= a.cols)
    return;
  CAT_AT(dst, row, col) = CAT_AT(a, row, col) + CAT_AT(b, row, col);
}

void cat_add(SField dst, const SField a, const SField b) {
  assert(a.rows == b.rows);
  assert(a.cols == b.cols);
  dim3 threads(32, 32);
  dim3 blocks((b.cols + threads.x - 1) / threads.x,
              (a.rows + threads.y - 1) / threads.y);
  hipLaunchKernelGGL(cat_add_kernel, blocks, threads, 0, 0, dst, a, b);
}

__global__ void cat_augment_kernel(SField dst, SField a, SField b, float2 c) {
  size_t row = blockIdx.y * blockDim.y + threadIdx.y;
  size_t col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= a.rows || col >= a.cols)
    return;
  CAT_AT(dst, row, col) = cmul(
      CAT_AT(a, row, col), make_float2(1, 0) + cmul(CAT_AT(b, row, col), c));
}

// a' = a + c(bxa)
void cat_augment(SField dst, const SField a, const SField b, const float2 c) {
  assert(a.rows == b.rows);
  assert(a.cols == b.cols);
  assert(a.rows == dst.rows);
  assert(a.cols == dst.cols);
  dim3 threads(32, 32);
  dim3 blocks((b.cols + threads.x - 1) / threads.x,
              (a.rows + threads.y - 1) / threads.y);
  hipLaunchKernelGGL(cat_augment_kernel, blocks, threads, 0, 0, dst, a, b, c);
}

__global__ void cat_scale_kernel(SField dst, const SField a, float2 c) {
  size_t row = blockIdx.y * blockDim.y + threadIdx.y;
  size_t col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= a.rows || col >= a.cols) {
    return;
  }
  CAT_AT(dst, row, col) = cmul(CAT_AT(a, row, col), c);
}

void cat_scale(SField dst, const SField a, float2 c) {
  dim3 threads(32, 32);
  dim3 blocks((a.cols + threads.x - 1) / threads.x,
              (a.rows + threads.y - 1) / threads.y);
  hipLaunchKernelGGL(cat_scale_kernel, blocks, threads, 0, 0, dst, a, c);
}

void cat_scale(SField dst, SField a, float c) {
  cat_scale(dst, a, make_float2(c, 0));
}

__global__ void cat_mult_kernel(SField dst, const SField a, const SField b) {
  size_t row = blockIdx.y * blockDim.y + threadIdx.y;
  size_t col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= a.rows || col >= a.cols) {
    return;
  }
  CAT_AT(dst, row, col) = cmul(CAT_AT(a, row, col), CAT_AT(b, row, col));
}

void cat_mult(SField dst, const SField a, const SField b) {
  dim3 threads(32, 32);
  dim3 blocks((a.cols + threads.x - 1) / threads.x,
              (a.rows + threads.y - 1) / threads.y);
  hipLaunchKernelGGL(cat_mult_kernel, blocks, threads, 0, 0, dst, a, b);
}

__global__ void cat_norm_kernel(SField dst, const SField a, const float c) {
  size_t row = blockIdx.y * blockDim.y + threadIdx.y;
  size_t col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= a.rows || col >= a.cols) {
    return;
  }
  float2 A = CAT_AT(a, row, col);
  CAT_AT(dst, row, col) = make_float2((A.x * A.x + A.y + A.y) * c, 0);
}

void cat_norm(SField dst, const SField a, float c) {
  dim3 threads(32, 32);
  dim3 blocks((a.cols + threads.x - 1) / threads.x,
              (a.rows + threads.y - 1) / threads.y);
  hipLaunchKernelGGL(cat_norm_kernel, blocks, threads, 0, 0, dst, a, c);
}

__global__ void cat_lap_kernel(SField lap, const SField u) {
  size_t row = blockIdx.y * blockDim.y + threadIdx.y;
  size_t col = blockIdx.x * blockDim.x + threadIdx.x;

  size_t trow = threadIdx.y + 1;
  size_t tcol = threadIdx.x + 1;

  __shared__ float2 t[34][34];
  const float2 ZERO = make_float2(0, 0);

  t[trow][tcol] = row < u.rows && col < u.cols ? CAT_AT(u, row, col) : ZERO;

  if (trow == 1)
    t[0][tcol] = row > 0 && tcol < u.cols ? CAT_AT(u, row - 1, col) : ZERO;
  if (trow == 32)
    t[33][tcol] =
        row <= u.rows && tcol < u.cols ? CAT_AT(u, row + 1, col) : ZERO;
  if (tcol == 1)
    t[trow][0] = col > 0 && trow < u.rows ? CAT_AT(u, row, col - 1) : ZERO;
  if (tcol == 32)
    t[trow][33] =
        col <= u.cols && trow < u.rows ? CAT_AT(u, row, col + 1) : ZERO;

  __syncthreads();

  if (row >= u.rows || col >= u.cols)
    return;
  if (row == 0 || col == 0 || row == u.rows - 1 || col == u.cols - 1) {
    CAT_AT(lap, row, col) = ZERO;
    return;
  }
  float2 v = -4.0f * t[trow][tcol] + t[trow - 1][tcol] + t[trow + 1][tcol] +
             t[trow][tcol + 1] + t[trow][tcol - 1];
  CAT_AT(lap, row, col) = v;
}

void cat_lap(SField lap, const SField u) {
  assert(u.rows == lap.rows);
  assert(u.cols == lap.cols);
  dim3 threads(32, 32);
  dim3 blocks((u.cols + threads.x - 1) / threads.x,
              (u.rows + threads.y - 1) / threads.y);
  hipLaunchKernelGGL(cat_lap_kernel, blocks, threads, 0, 0, lap, u);
}

constexpr float2 ZERO = {0, 0};

__global__ void cat_filter_kernel(SField m) {
  size_t row = blockIdx.y * blockDim.y + threadIdx.y;
  size_t col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= m.rows || col >= m.cols)
    return;
  float2 f2 = CAT_AT(m, row, col);
  if (f2.x * f2.x + f2.y * f2.y > 0.00f)
    return;
  CAT_AT(m, row, col) = ZERO;
}

void cat_filter(SField m) {
  dim3 threads(32, 32);
  dim3 blocks((m.cols + threads.x - 1) / threads.x,
              (m.rows + threads.y - 1) / threads.y);
  hipLaunchKernelGGL(cat_filter_kernel, blocks, threads, 0, 0, m);
}

__global__ void cat_step_kernel(SField next, const SField t, const SField u,
                                const SField v, float dt) {
  size_t row = blockIdx.y * blockDim.y + threadIdx.y;
  size_t col = blockIdx.x * blockDim.x + threadIdx.x;

  size_t trow = threadIdx.y + 1;
  size_t tcol = threadIdx.x + 1;

  __shared__ float2 s[34][34];

  s[trow][tcol] = row < u.rows && col < u.cols ? CAT_AT(u, row, col) : ZERO;

  if (trow == 1)
    s[0][tcol] = row > 0 && col < u.cols ? CAT_AT(u, row - 1, col) : ZERO;
  if (trow == 32)
    s[33][tcol] =
        row + 1 < u.rows && col < u.cols ? CAT_AT(u, row + 1, col) : ZERO;
  if (tcol == 1)
    s[trow][0] = col > 0 && row < u.rows ? CAT_AT(u, row, col - 1) : ZERO;
  if (tcol == 32)
    s[trow][33] =
        col + 1 < u.cols && row < u.rows ? CAT_AT(u, row, col + 1) : ZERO;

  if (trow == 1 && tcol == 1)
    s[0][0] = row > 0 && col > 0 ? CAT_AT(u, row - 1, col - 1) : ZERO;
  if (trow == 1 && tcol == 32)
    s[0][33] = row > 0 && col + 1 < u.cols ? CAT_AT(u, row - 1, col + 1) : ZERO;
  if (trow == 32 && tcol == 1)
    s[33][0] = col > 0 && row + 1 < u.rows ? CAT_AT(u, row + 1, col - 1) : ZERO;
  if (trow == 32 && tcol == 32)
    s[33][33] = col + 1 < u.cols && row + 1 < u.rows
                    ? CAT_AT(u, row + 1, col + 1)
                    : ZERO;

  __syncthreads();

  if (row >= u.rows || col >= u.cols)
    return;
  if (row == 0 || col == 0 || row == u.rows - 1 || col == u.cols - 1) {
    CAT_AT(next, row, col) = ZERO;
    return;
  }

  const float2 wave = s[trow][tcol];
  const float2 edges = (s[trow - 1][tcol] + s[trow + 1][tcol] +
                        s[trow][tcol + 1] + s[trow][tcol - 1]);
  const float2 corners = (s[trow - 1][tcol - 1] + s[trow + 1][tcol - 1] +
                          s[trow - 1][tcol + 1] + s[trow + 1][tcol + 1]);
  const float2 lap = (-20.0f * wave + 4.0f * edges + corners) / 6.0f;
  // const float2 lap = -4.0f * wave + edges;

  const float2 pot = CAT_AT(v, row, col);
  const float2 wave_coef = make_float2(-2 * pot.y * dt, -2 * -pot.x * dt);
  const float2 lap_coef = make_float2(0, dt);

  // |u,t+dt> = |u,t-dt> - 2idtH |u,t>
  // |u,t+dt> = |u,t-dt> - 2idt(V - H0) |u,t>
  // |u,t+dt> = |u,t-dt> - 2idt(V - lap/2) |u,t>
  // |u,t+dt> = |u,t-dt> - 2idtV |u,t> + idtlap |u,t>
  CAT_AT(next, row, col) =
      CAT_AT(t, row, col) + cmul(wave_coef, wave) + cmul(lap_coef, lap);
}

void cat_step(SField next, const SField t, const SField u, const SField v,
              const float dt) {
  assert(u.rows == next.rows);
  assert(u.cols == next.cols);
  dim3 threads(32, 32);
  dim3 blocks((u.cols + threads.x - 1) / threads.x,
              (u.rows + threads.y - 1) / threads.y);
  hipLaunchKernelGGL(cat_step_kernel, blocks, threads, 0, 0, next, t, u, v, dt);
}

__global__ void cat_grad_kernel(SField X, SField Y, const SField u) {
  size_t row = blockIdx.y * blockDim.y + threadIdx.y;
  size_t col = blockIdx.x * blockDim.x + threadIdx.x;

  size_t trow = threadIdx.y + 1;
  size_t tcol = threadIdx.x + 1;

  __shared__ float2 x[34][34];
  __shared__ float2 y[34][34];
  const float2 ZERO = make_float2(0, 0);

  x[trow][tcol] = row < u.rows && col < u.cols ? CAT_AT(u, row, col) : ZERO;
  y[trow][tcol] = row < u.rows && col < u.cols ? CAT_AT(u, row, col) : ZERO;

  if (trow == 1)
    y[0][tcol] = row > 0 && tcol < u.cols ? CAT_AT(u, row - 1, col) : ZERO;
  if (trow == 32)
    y[33][tcol] =
        row <= u.rows && tcol < u.cols ? CAT_AT(u, row + 1, col) : ZERO;
  if (tcol == 1)
    x[trow][0] = col > 0 && trow < u.rows ? CAT_AT(u, row, col - 1) : ZERO;
  if (tcol == 32)
    x[trow][33] =
        col <= u.cols && trow < u.rows ? CAT_AT(u, row, col + 1) : ZERO;

  __syncthreads();

  if (row >= u.rows || col >= u.cols)
    return;
  if (row == 0 || col == 0 || row == u.rows - 1 || col == u.cols - 1) {
    CAT_AT(X, row, col) = ZERO;
    CAT_AT(Y, row, col) = ZERO;
    return;
  }
  CAT_AT(X, row, col) = 0.5f * (-x[trow][tcol - 1] + x[trow][tcol + 1]);
  CAT_AT(Y, row, col) = 0.5f * (-y[trow - 1][tcol] + y[trow + 1][tcol]);
}

void cat_grad(VField dst, const SField a) {
  assert(a.rows == dst.x.rows);
  assert(a.cols == dst.x.cols);
  assert(a.rows == dst.y.rows);
  assert(a.cols == dst.y.cols);
  dim3 threads(32, 32);
  dim3 blocks((a.cols + threads.x - 1) / threads.x,
              (a.rows + threads.y - 1) / threads.y);
  hipLaunchKernelGGL(cat_grad_kernel, blocks, threads, 0, 0, dst.x, dst.y, a);
}

__global__ void cfd_div_kernel(SField div, const VField a) {
  const size_t rows = a.x.rows;
  const size_t cols = a.x.cols;
  size_t row = blockIdx.y * blockDim.y + threadIdx.y;
  size_t col = blockIdx.x * blockDim.x + threadIdx.x;

  size_t trow = threadIdx.y + 1;
  size_t tcol = threadIdx.x + 1;

  __shared__ float2 x[34][34];
  __shared__ float2 y[34][34];
  const float2 ZERO = make_float2(0, 0);

  x[trow][tcol] = row < rows && col < cols ? CAT_AT(a.x, row, col) : ZERO;
  y[trow][tcol] = row < rows && col < cols ? CAT_AT(a.y, row, col) : ZERO;

  if (trow == 1)
    y[0][tcol] = row > 0 && tcol < cols ? CAT_AT(a.y, row - 1, col) : ZERO;
  if (trow == 32)
    y[33][tcol] = row <= rows && tcol < cols ? CAT_AT(a.y, row + 1, col) : ZERO;
  if (tcol == 1)
    x[trow][0] = col > 0 && trow < rows ? CAT_AT(a.x, row, col - 1) : ZERO;
  if (tcol == 32)
    x[trow][33] = col <= cols && trow < rows ? CAT_AT(a.x, row, col + 1) : ZERO;

  __syncthreads();

  if (row >= rows || col >= cols)
    return;
  if (row == 0 || col == 0 || row == rows - 1 || col == cols - 1) {
    CAT_AT(div, row, col) = ZERO;
    CAT_AT(div, row, col) = ZERO;
    return;
  }
  float2 v = -y[trow - 1][tcol] + y[trow + 1][tcol] + x[trow][tcol + 1] -
             x[trow][tcol - 1];
  CAT_AT(div, row, col) = 0.5f * v;
}

void cfd_div(SField dst, const VField a) {
  assert(dst.rows == a.x.rows);
  assert(dst.cols == a.x.cols);
  assert(dst.rows == a.y.rows);
  assert(dst.cols == a.y.cols);
  dim3 threads(32, 32);
  dim3 blocks((dst.cols + threads.x - 1) / threads.x,
              (dst.rows + threads.y - 1) / threads.y);
  hipLaunchKernelGGL(cfd_div_kernel, blocks, threads, 0, 0, dst, a);
}

__global__ void cfd_dot_kernel(SField dot, const VField a, const VField b) {
  const size_t rows = dot.rows;
  const size_t cols = dot.cols;
  size_t row = blockIdx.y * blockDim.y + threadIdx.y;
  size_t col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row >= dot.rows || col >= dot.cols)
    return;
  CAT_AT(dot, row, col) = cmul(CAT_AT(a.x, row, col), CAT_AT(b.x, row, col)) +
                          cmul(CAT_AT(a.y, row, col), CAT_AT(b.y, row, col));
}

void cfd_dot(SField dst, const VField a, const VField b) {
  assert(dst.rows == a.x.rows);
  assert(dst.cols == a.x.cols);
  assert(dst.rows == a.y.rows);
  assert(dst.cols == a.y.cols);
  assert(dst.rows == b.x.rows);
  assert(dst.cols == b.x.cols);
  assert(dst.rows == b.y.rows);
  assert(dst.cols == b.y.cols);

  dim3 threads(32, 32);
  dim3 blocks((dst.cols + threads.x - 1) / threads.x,
              (dst.rows + threads.y - 1) / threads.y);
  hipLaunchKernelGGL(cfd_dot_kernel, blocks, threads, 0, 0, dst, a, b);
}
