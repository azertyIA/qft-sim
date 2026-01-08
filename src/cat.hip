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

dim3 threads(32, 32);

__device__ __host__ __forceinline__ float2 cmul(float2 a, float2 b) {
  return {fmaf(a.x, b.x, -a.y * b.y), fmaf(a.x, b.y, a.y * b.x)};
}

__device__ __host__ __forceinline__ float2 cmac(float2 a, float2 b, float2 c) {
  return {fmaf(a.x, b.x, fmaf(-a.y, b.y, c.x)),
          fmaf(a.x, b.y, fmaf(a.y, b.x, c.y))};
}

SField s_alloc_host(size_t rows, size_t cols) {
  SField m;
  m.rows = rows;
  m.cols = cols;
  m.stride = cols;
  m.data = (float2 *)malloc(rows * cols * sizeof(float2));
  return m;
}

SField s_adapt(SField h) {
  SField m;
  m.rows = h.rows;
  m.cols = h.cols;
  m.stride = h.cols;
  HIP_CHECK(hipMalloc(&m.data, h.rows * h.cols * sizeof(float2)));
  return m;
}

void s_free_host(SField m) { free(m.data); }

void s_free_device(SField m) { HIP_CHECK(hipFree(m.data)); }

void s_preview(SField m) {
  for (int r = 0; r < 4 && r < m.rows; ++r) {
    for (int c = 0; c < 4 && c < m.cols; ++c) {
      float2 v = CAT_AT(m, r, c);
      printf("(%.2f + %.2fi)\t", v.x, v.y);
    }
    printf("\n");
  }
  printf("\n");
}

void s_to_device(SField d, SField h) {
  size_t total = d.rows * d.cols;
  HIP_CHECK(
      hipMemcpy(d.data, h.data, total * sizeof(float2), hipMemcpyHostToDevice));
}

void s_to_host(SField h, SField d) {
  size_t total = d.rows * d.cols;
  HIP_CHECK(
      hipMemcpy(h.data, d.data, total * sizeof(float2), hipMemcpyDeviceToHost));
}

void s_host_fill(SField dst, SFieldFn f) {
  for (size_t row = 0; row < dst.rows; row++) {
    for (size_t col = 0; col < dst.cols; col++) {
      CAT_AT(dst, row, col) = f(col, row);
    }
  }
}

void s_clone(SField dst, const SField src) {
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

void s_fill(SField m, float2 c) {
  dim3 blocks((m.cols + threads.x - 1) / threads.x,
              (m.rows + threads.y - 1) / threads.y);
  hipLaunchKernelGGL(cat_fill_kernel, blocks, threads, 0, 0, m, c);
}

void s_fill(SField m, float c) { s_fill(m, make_float2(c, 0)); }

__global__ void cat_block_kernel(SField m, const float2 v, const int2 x,
                                 const int2 y) {
  size_t row = blockIdx.y * blockDim.y + threadIdx.y;
  size_t col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= y.y || row <= y.x || col >= x.y || col <= x.x)
    return;
  CAT_AT(m, row, col) += v;
}

void s_block(SField m, const float2 v, const int2 x, const int2 y) {
  dim3 blocks((m.cols + threads.x - 1) / threads.x,
              (m.rows + threads.y - 1) / threads.y);
  hipLaunchKernelGGL(cat_block_kernel, blocks, threads, 0, 0, m, v, x, y);
}

__global__ void cat_add_kernel(SField dst, SField a, SField b) {
  size_t row = blockIdx.y * blockDim.y + threadIdx.y;
  size_t col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= a.rows || col >= a.cols)
    return;
  // CAT_AT(dst, row, col) = CAT_AT(a, row, col) + CAT_AT(b, row, col);
  float2 A = CAT_AT(a, row, col);
  float2 B = CAT_AT(b, row, col);
  float2 *D = &CAT_AT(dst, row, col);
  D->x = A.x + B.x;
  D->y = A.y + B.y;
}

void s_add(SField dst, const SField a, const SField b) {
  assert(a.rows == b.rows);
  assert(a.cols == b.cols);
  dim3 blocks((b.cols + threads.x - 1) / threads.x,
              (a.rows + threads.y - 1) / threads.y);
  hipLaunchKernelGGL(cat_add_kernel, blocks, threads, 0, 0, dst, a, b);
}

__global__ void s_scale_add_kernel(SField dst, SField a, SField b, float2 c) {
  size_t row = blockIdx.y * blockDim.y + threadIdx.y;
  size_t col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= a.rows || col >= a.cols)
    return;
  float2 *D = &CAT_AT(dst, row, col);
  float2 B = CAT_AT(a, row, col);
  float2 A = CAT_AT(b, row, col);
  D->x = fmaf(A.x, c.x, fmaf(-A.y, c.y, B.x));
  D->y = fmaf(A.x, c.y, fmaf(A.y, c.x, B.y));
  // CAT_AT(dst, row, col) = CAT_AT(a, row, col) + cmul(CAT_AT(b, row, col), c);
}

void s_scale_add(SField dst, const SField a, const SField b, const float2 c) {
  assert(a.rows == b.rows);
  assert(a.cols == b.cols);
  assert(a.rows == dst.rows);
  assert(a.cols == dst.cols);
  dim3 blocks((b.cols + threads.x - 1) / threads.x,
              (a.rows + threads.y - 1) / threads.y);
  hipLaunchKernelGGL(s_scale_add_kernel, blocks, threads, 0, 0, dst, a, b, c);
}

__global__ void cat_scale_kernel(SField dst, const SField a, float2 c) {
  size_t row = blockIdx.y * blockDim.y + threadIdx.y;
  size_t col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= a.rows || col >= a.cols) {
    return;
  }
  // CAT_AT(dst, row, col) = cmul(CAT_AT(a, row, col), c);
  float2 *D = &CAT_AT(dst, row, col);
  float2 A = CAT_AT(a, row, col);
  D->x = fmaf(A.x, c.x, -A.y * c.y);
  D->y = fmaf(A.x, c.y, A.y * c.x);
}

void s_scale(SField dst, const SField a, float2 c) {
  dim3 threads(32, 32);
  dim3 blocks((a.cols + threads.x - 1) / threads.x,
              (a.rows + threads.y - 1) / threads.y);
  hipLaunchKernelGGL(cat_scale_kernel, blocks, threads, 0, 0, dst, a, c);
}

void s_scale(SField dst, SField a, float c) {
  s_scale(dst, a, make_float2(c, 0));
}

__global__ void cat_mult_kernel(SField dst, const SField a, const SField b) {
  size_t row = blockIdx.y * blockDim.y + threadIdx.y;
  size_t col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= a.rows || col >= a.cols) {
    return;
  }

  float2 *D = &CAT_AT(dst, row, col);
  float2 A = CAT_AT(a, row, col);
  float2 B = CAT_AT(b, row, col);
  D->x = fmaf(B.x, A.x, -B.y * A.y);
  D->y = fmaf(B.x, A.y, B.y * A.x);
  // CAT_AT(dst, row, col) = cmul(CAT_AT(a, row, col), CAT_AT(b, row, col));
}

void s_mult(SField dst, const SField a, const SField b) {
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

void s_norm(SField dst, const SField a, float c) {
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
  float2 *tmp = &CAT_AT(u, row, col);

  if (row < u.rows && col < u.cols) {
    t[trow][tcol].x = tmp->x;
    t[trow][tcol].y = tmp->y;
  } else {
    t[trow][tcol].x = 0;
    t[trow][tcol].y = 0;
  }

  // shared memory is good at assigning float2s.
  if (trow == 1 && row > 0 && tcol < u.cols) {
    t[0][tcol] = *(tmp - u.stride);
  } else if (trow == 32 && row <= u.rows && tcol < u.cols) {
    t[33][tcol] = *(tmp + u.stride);
  }

  if (tcol == 1 && col > 0 && trow < u.rows) {
    t[trow][0] = *(tmp - 1);
  } else if (tcol == 32) {
    t[trow][33] = col <= u.cols && trow < u.rows ? *(tmp + 1) : ZERO;
  }

  __syncthreads();

  // always good to render the things, and then assign by component.
  tmp = &CAT_AT(u, row, col);
  if (row >= u.rows || col >= u.cols)
    return;
  if (row == 0 || col == 0 || row == u.rows - 1 || col == u.cols - 1) {
    CAT_AT(lap, row, col).x = 0;
    CAT_AT(lap, row, col).y = 0;
    return;
  }
  CAT_AT(lap, row, col).x = -4.0f * t[trow][tcol].x + t[trow - 1][tcol].x +
                            t[trow + 1][tcol].x + t[trow][tcol + 1].x +
                            t[trow][tcol - 1].x;
  CAT_AT(lap, row, col).y = -4.0f * t[trow][tcol].y + t[trow - 1][tcol].y +
                            t[trow + 1][tcol].y + t[trow][tcol + 1].y +
                            t[trow][tcol - 1].y;
}

void s_lap(SField lap, const SField u) {
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

void s_filter(SField m) {
  dim3 blocks((m.cols + threads.x - 1) / threads.x,
              (m.rows + threads.y - 1) / threads.y);
  hipLaunchKernelGGL(cat_filter_kernel, blocks, threads, 0, 0, m);
}

__global__ void cat_grad_kernel(SField X, SField Y, const SField u) {
  size_t row = blockIdx.y * blockDim.y + threadIdx.y;
  size_t col = blockIdx.x * blockDim.x + threadIdx.x;

  size_t trow = threadIdx.y + 1;
  size_t tcol = threadIdx.x + 1;

  __shared__ float2 t[34][34];
  const float2 ZERO = make_float2(0, 0);

  t[trow][tcol] = row < u.rows && col < u.cols ? CAT_AT(u, row, col) : ZERO;

  if (trow == 1)
    t[0][tcol] = row > 0 && tcol < u.cols ? CAT_AT(u, row - 1, col) : ZERO;
  else if (trow == 32)
    t[33][tcol] =
        row <= u.rows && tcol < u.cols ? CAT_AT(u, row + 1, col) : ZERO;

  if (tcol == 1)
    t[trow][0] = col > 0 && trow < u.rows ? CAT_AT(u, row, col - 1) : ZERO;
  else if (tcol == 32)
    t[trow][33] =
        col <= u.cols && trow < u.rows ? CAT_AT(u, row, col + 1) : ZERO;

  __syncthreads();

  if (row >= u.rows || col >= u.cols)
    return;
  CAT_AT(X, row, col) = 0.5f * (-t[trow][tcol - 1] + t[trow][tcol + 1]);
  CAT_AT(Y, row, col) = 0.5f * (-t[trow - 1][tcol] + t[trow + 1][tcol]);
  if (row == 0 || col == 0 || row == u.rows - 1 || col == u.cols - 1) {
    CAT_AT(X, row, col) = ZERO;
    CAT_AT(Y, row, col) = ZERO;
    return;
  }
}

void s_grad(VField dst, const SField a) {
  assert(a.rows == dst.x.rows);
  assert(a.cols == dst.x.cols);
  assert(a.rows == dst.y.rows);
  assert(a.cols == dst.y.cols);
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
  float2 *tmp = &CAT_AT(a.x, row, col);
  x[trow][tcol] = row < rows && col < cols ? *(tmp) : ZERO;

  if (tcol == 1 && col > 0 && trow < rows)
    x[trow][0] = CAT_AT(a.x, row, col - 1);
  else if (tcol == 32 && col <= cols && trow < rows)
    x[trow][33] = CAT_AT(a.x, row, col + 1);

  tmp = &CAT_AT(a.y, row, col);
  y[trow][tcol] = row < rows && col < cols ? CAT_AT(a.y, row, col) : ZERO;

  if (trow == 1 && row > 0 && tcol < cols)
    y[0][tcol] = CAT_AT(a.y, row - 1, col);
  else if (trow == 32 && row <= rows && tcol < cols)
    y[33][tcol] = CAT_AT(a.y, row + 1, col);

  __syncthreads();

  tmp = &CAT_AT(div, row, col);
  if (row >= rows || col >= cols)
    return;
  if (row == 0 || col == 0 || row == rows - 1 || col == cols - 1) {
    tmp->x = 0;
    tmp->y = 0;
    return;
  }
  float2 v = -y[trow - 1][tcol] + y[trow + 1][tcol] + x[trow][tcol + 1] -
             x[trow][tcol - 1];
  tmp->x = 0.5f * v.x;
  tmp->y = 0.5f * v.y;
}

void v_div(SField dst, const VField a) {
  assert(dst.rows == a.x.rows);
  assert(dst.cols == a.x.cols);
  assert(dst.rows == a.y.rows);
  assert(dst.cols == a.y.cols);
  dim3 blocks((dst.cols + threads.x - 1) / threads.x,
              (dst.rows + threads.y - 1) / threads.y);
  hipLaunchKernelGGL(cfd_div_kernel, blocks, threads, 0, 0, dst, a);
}

__global__ void cfd_dot_kernel(SField dot, const VField a, const VField b) {
  const size_t rows = dot.rows;
  const size_t cols = dot.cols;
  size_t row = blockIdx.y * blockDim.y + threadIdx.y;
  size_t col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row >= rows || col >= cols)
    return;
  CAT_AT(dot, row, col) = cmul(CAT_AT(a.x, row, col), CAT_AT(b.x, row, col)) +
                          cmul(CAT_AT(a.y, row, col), CAT_AT(b.y, row, col));
}

void v_dot(SField dst, const VField a, const VField b) {
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

__global__ void f_dot_grad_kernel(SField dot, const VField a, const SField u) {
  const size_t rows = dot.rows;
  const size_t cols = dot.cols;

  size_t row = blockIdx.y * blockDim.y + threadIdx.y;
  size_t col = blockIdx.x * blockDim.x + threadIdx.x;

  size_t trow = threadIdx.y + 1;
  size_t tcol = threadIdx.x + 1;

  __shared__ float2 x[34][34];
  __shared__ float2 y[34][34];
  const float2 ZERO = make_float2(0, 0);

  x[trow][tcol] = row < rows && col < cols ? CAT_AT(u, row, col) : ZERO;
  y[trow][tcol] = row < rows && col < cols ? CAT_AT(u, row, col) : ZERO;

  if (trow == 1)
    y[0][tcol] = row > 0 && tcol < cols ? CAT_AT(u, row - 1, col) : ZERO;
  if (trow == 32)
    y[33][tcol] = row <= rows && tcol < cols ? CAT_AT(u, row + 1, col) : ZERO;
  if (tcol == 1)
    x[trow][0] = col > 0 && trow < rows ? CAT_AT(u, row, col - 1) : ZERO;
  if (tcol == 32)
    x[trow][33] = col <= cols && trow < rows ? CAT_AT(u, row, col + 1) : ZERO;

  __syncthreads();

  if (row >= rows || col >= cols)
    return;
  if (row == 0 || col == 0 || row == rows - 1 || col == cols - 1) {
    CAT_AT(dot, row, col) = ZERO;
    return;
  }

  CAT_AT(dot, row, col) =
      0.5f *
      (cmul((-x[trow][tcol - 1] + x[trow][tcol + 1]), CAT_AT(a.x, row, col)) +
       cmul((-y[trow - 1][tcol] + y[trow + 1][tcol]), CAT_AT(a.y, row, col)));
}

void v_dot_grad(SField dst, const VField a, const SField u) {
  assert(dst.rows == a.x.rows);
  assert(dst.cols == a.x.cols);
  assert(dst.rows == a.y.rows);
  assert(dst.cols == a.y.cols);
  assert(dst.rows == u.rows);
  assert(dst.cols == u.cols);
  dim3 blocks((dst.cols + threads.x - 1) / threads.x,
              (dst.rows + threads.y - 1) / threads.y);
  hipLaunchKernelGGL(f_dot_grad_kernel, blocks, threads, 0, 0, dst, a, u);
}

__global__ void s_exp_kernel(SField dst, const SField a) {
  size_t row = blockIdx.y * blockDim.y + threadIdx.y;
  size_t col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= a.rows || col >= a.cols) {
    return;
  }

  float2 *D = &CAT_AT(dst, row, col);
  float2 A = CAT_AT(a, row, col);
  D->x = exp(A.x) * cos(A.y);
  D->y = exp(A.x) * sin(A.y);
  // CAT_AT(dst, row, col) = cmul(CAT_AT(a, row, col), CAT_AT(b, row, col));
}

void s_exp(SField dst, const SField a) {
  assert(dst.rows == a.rows);
  assert(dst.cols == a.cols);
  dim3 blocks((dst.cols + threads.x - 1) / threads.x,
              (dst.rows + threads.y - 1) / threads.y);
  hipLaunchKernelGGL(s_exp_kernel, blocks, threads, 0, 0, dst, a);
};

__global__ void q_step_so_kernel(SField next, const SField curr,
                                 const SField prev, const SField h,
                                 const VField a, const float2 v2_factor,
                                 const float2 h_factor) {
  // kernel init and indexing takes 1.6 ms/100 steps
  const size_t rows = next.rows;
  const size_t cols = next.cols;

  size_t trow = threadIdx.y + 1;
  size_t tcol = threadIdx.x + 1;

  size_t row = blockIdx.y * blockDim.y + trow - 1;
  size_t col = blockIdx.x * blockDim.x + tcol - 1;

  __shared__ float2 t[34][34];
  float2 H;
  // pointer trick saves 0.1 ms/100 steps
  float2 *tmp = &CAT_AT(curr, row, col);

  // filling up t takes 0.6 ms/100 steps
  if (row < rows && col < cols)
    t[trow][tcol] = *tmp;

  if (trow == 1 && row > 0)
    t[0][tcol] = *(tmp - curr.stride);
  else if (trow == 32 && row <= rows)
    t[33][tcol] = *(tmp + curr.stride);
  if (tcol == 1 && col > 0)
    t[trow][0] = *(tmp - 1);
  else if (tcol == 32 && col <= cols)
    t[trow][33] = *(tmp + 1);

  __syncthreads();

  // adding this pointer costs 0.1 ms
  // tmp = &CAT_AT(prev, row, col);

  if (row >= rows || col >= cols)
    return;
  if (row == 0 || col == 0 || row == rows - 1 || col == cols - 1) {
    CAT_AT(next, row, col) = ZERO;
    return;
  }

  // calculating next takes 2.6 ms/100 steps
  // first term costs 0.4 ms/100 steps
  H = -4.0f * t[trow][tcol] + t[trow - 1][tcol] + t[trow + 1][tcol] +
      t[trow][tcol + 1] + t[trow][tcol - 1];

  H = cmul(v2_factor, H);
  H += 0.5f *
       (cmul((-t[trow][tcol - 1] + t[trow][tcol + 1]), CAT_AT(a.x, row, col)) +
        cmul((-t[trow - 1][tcol] + t[trow + 1][tcol]), CAT_AT(a.y, row, col)));
  H += cmul(CAT_AT(curr, row, col), CAT_AT(h, row, col));
  H = cmul(h_factor, H);

  // setting next takes 0.1 ms/100 steps
  // component-wise saves 0.1 ms
  // tmp->x = CAT_AT(prev, row, col).x + H.x;
  // tmp->y = CAT_AT(prev, row, col).y + H.y;
  // CAT_AT(next, row, col).x = tmp->x + H.x;
  // CAT_AT(next, row, col).y = tmp->y + H.y;
  CAT_AT(next, row, col).x = CAT_AT(prev, row, col).x + H.x;
  CAT_AT(next, row, col).y = CAT_AT(prev, row, col).y + H.y;
}

void q_step_so(SField next, const SField curr, const SField prev,
               const SField h, const VField a, const float2 v2_factor,
               const float2 h_factor) {
  dim3 blocks((next.cols + threads.x - 1) / threads.x,
              (next.rows + threads.y - 1) / threads.y);
  hipLaunchKernelGGL(q_step_so_kernel, blocks, threads, 0, 0, next, curr, prev,
                     h, a, v2_factor, h_factor);
}

__global__ void cg_step_so_kernel(const SField curr, SField prev,
                                  const VField Tr, const VField Td,
                                  const SField O, const float d2_factor,
                                  const float h_factor) {
  const size_t rows = curr.rows;
  const size_t cols = curr.cols;

  size_t trow = threadIdx.y + 1;
  size_t tcol = threadIdx.x + 1;

  size_t row = blockIdx.y * blockDim.y + trow - 1;
  size_t col = blockIdx.x * blockDim.x + tcol - 1;

  __shared__ float2 u[34][34];
  float2 *tmp = &CAT_AT(curr, row, col);

  if (row < rows && col < cols)
    u[trow][tcol] = *tmp;
  if (trow == 1 && row > 0)
    u[0][tcol] = *(tmp - curr.stride);
  else if (trow == 32 && row < rows - 1)
    u[33][tcol] = *(tmp + curr.stride);
  if (tcol == 1 && col > 0)
    u[trow][0] = *(tmp - 1);
  else if (tcol == 32 && col < cols - 1)
    u[trow][33] = *(tmp + 1);

  __syncthreads();

  if (row >= rows || col >= cols)
    return;
  if (row == 0 || col == 0 || row == rows - 1 || col == cols - 1) {
    CAT_AT(prev, row, col) = ZERO;
    return;
  }

  float2 H = cmac(CAT_AT(Td.y, row - 1, col), u[trow - 1][tcol],
                  cmac(CAT_AT(Tr.y, row, col), u[trow + 1][tcol],
                       cmac(CAT_AT(Tr.x, row, col), u[trow][tcol + 1],
                            cmac(CAT_AT(Td.x, row, col - 1), u[trow][tcol - 1],
                                 cmul(CAT_AT(O, row, col), u[trow][tcol])))));

  CAT_AT(prev, row, col).x -= H.y * h_factor;
  CAT_AT(prev, row, col).y += H.x * h_factor;
}

void cg_step_so(const SField curr, const SField prev, const VField Tr,
                const VField Td, const SField O, const float d2_factor,
                const float h_factor) {
  dim3 blocks((curr.cols + threads.x - 1) / threads.x,
              (curr.rows + threads.y - 1) / threads.y);
  hipLaunchKernelGGL(cg_step_so_kernel, blocks, threads, 0, 0, curr, prev, Tr,
                     Td, O, d2_factor, h_factor);
}
