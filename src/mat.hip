#include "mat.h"
#include <assert.h>
#include <cstddef>
#include <cstdio>
#include <stdlib.h>

#define HIP_CHECK(command)                                                     \
  {                                                                            \
    hipError_t status = command;                                               \
    assert(status == hipSuccess);                                              \
  }

Matrix mat_alloc_host(size_t rows, size_t cols) {
  Matrix m;
  m.rows = rows;
  m.cols = cols;
  m.stride = cols;
  m.data = (float *)malloc(rows * cols * sizeof(float));
  return m;
}

Matrix mat_alloc_device(size_t rows, size_t cols) {
  Matrix m;
  m.rows = rows;
  m.cols = cols;
  m.stride = cols;
  HIP_CHECK(hipMalloc(&m.data, rows * cols * sizeof(float)));
  return m;
}

Matrix mat_adapt(const Matrix m) {
  Matrix d = mat_alloc_device(m.rows, m.cols);
  return d;
}

void mat_free_host(Matrix m) { free(m.data); }

void mat_free_device(Matrix m) { HIP_CHECK(hipFree(m.data)); }

void mat_copy_to_device(Matrix d, Matrix h) {
  size_t total = d.rows * d.cols;
  HIP_CHECK(
      hipMemcpy(d.data, h.data, total * sizeof(float), hipMemcpyHostToDevice));
}

void mat_copy_to_host(Matrix h, Matrix d) {
  size_t total = d.rows * d.cols;
  HIP_CHECK(
      hipMemcpy(h.data, d.data, total * sizeof(float), hipMemcpyDeviceToHost));
}

void mat_preview(Matrix m) {
  for (size_t row = 0; row < 8 && row < m.rows; row++) {
    for (size_t col = 0; col < 8 && col < m.cols; col++) {
      printf("%.1f\t", MAT_AT(m, row, col));
    }
    printf("\n");
  }
  printf("\n");
}

void mat_clone_device(Matrix dst, const Matrix a) {
  assert(dst.rows == a.rows);
  assert(dst.cols == a.cols);
  size_t total = dst.rows * dst.cols;
  HIP_CHECK(hipMemcpy(dst.data, a.data, total * sizeof(float),
                      hipMemcpyDeviceToDevice));
}

void mat_fill_host(Matrix m, float value) {
  for (int row = 0; row < m.rows; row++) {
    for (int col = 0; col < m.cols; col++) {
      MAT_AT(m, row, col) = value;
    }
  }
}

__global__ void mat_fill_kernel(float *m, float value, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    m[idx] = value;
  }
}

void mat_fill_device(Matrix m, float value) {
  int total = m.rows * m.cols;
  dim3 threads(256);
  dim3 blocks((total + threads.x - 1) / threads.x);
  hipLaunchKernelGGL(mat_fill_kernel, blocks, threads, 0, 0, m.data, value,
                     total);
}

void mat_add_host(Matrix dst, const Matrix a, const Matrix b) {
  assert(a.rows == b.rows && a.rows == dst.rows);
  assert(a.cols == b.cols && a.cols == dst.cols);
  for (int row = 0; row < a.rows; row++) {
    for (int col = 0; col < a.cols; col++) {
      MAT_AT(dst, row, col) = MAT_AT(a, row, col) + MAT_AT(b, row, col);
    }
  }
}

__global__ void mat_add_kernel(Matrix dst, Matrix a, Matrix b) {
  size_t row = blockIdx.y * blockDim.y + threadIdx.y;
  size_t col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= a.rows || col >= a.cols) {
    return;
  }
  MAT_AT(dst, row, col) = MAT_AT(a, row, col) + MAT_AT(b, row, col);
}

void mat_add_device(Matrix dst, const Matrix a, const Matrix b) {
  assert(a.rows == b.rows);
  assert(a.cols == b.cols);
  dim3 threads(32, 32);
  dim3 blocks((b.cols + threads.x - 1) / threads.x,
              (a.rows + threads.y - 1) / threads.y);
  hipLaunchKernelGGL(mat_add_kernel, blocks, threads, 0, 0, dst, a, b);
}

void mat_mult_host(Matrix dst, const Matrix a, const Matrix b) {
  assert(a.cols == b.rows);
  assert(a.rows == dst.rows);
  assert(b.cols == dst.cols);

  for (size_t row = 0; row < a.rows; row++) {
    for (size_t col = 0; col < b.cols; col++) {
      for (size_t dot = 0; dot < a.cols; dot++) {
        MAT_AT(dst, row, col) += MAT_AT(a, row, dot) * MAT_AT(b, dot, col);
      }
    }
  }
}

__global__ void mat_mult_kernel(Matrix dst, Matrix a, Matrix b) {
  size_t row = blockIdx.y * blockDim.y + threadIdx.y;
  size_t col = blockIdx.x * blockDim.x + threadIdx.x;

  __shared__ float sA[32][32];
  __shared__ float sB[32][32];

  float sum = 0;
  size_t tiles = (a.cols + 32 - 1) / 32;

  for (int tile = 0; tile < tiles; tile++) {
    int kA = tile * 32 + threadIdx.x;
    int kB = tile * 32 + threadIdx.y;

    sA[threadIdx.y][threadIdx.x] =
        (row < dst.rows && kA < a.cols) ? MAT_AT(a, row, kA) : 0.0f;
    sB[threadIdx.y][threadIdx.x] =
        (kB < a.cols && col < dst.cols) ? MAT_AT(b, kB, col) : 0.0f;

    __syncthreads();

    for (int k = 0; k < 32; k++) {
      sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
    }

    __syncthreads();
  }

  if (row >= dst.rows || col >= dst.cols) {
    return;
  }

  MAT_AT(dst, row, col) = sum;
}

void mat_mult_device(Matrix dst, const Matrix a, const Matrix b) {
  assert(a.cols == b.rows);
  assert(a.rows == dst.rows);
  assert(b.cols == dst.cols);
  dim3 threads(32, 32);
  dim3 blocks((b.cols + threads.x - 1) / threads.x,
              (a.rows + threads.y - 1) / threads.y);
  hipLaunchKernelGGL(mat_mult_kernel, blocks, threads, 0, 0, dst, a, b);
  HIP_CHECK(hipDeviceSynchronize());
}

__global__ void mat_scale_kernel(Matrix dst, const Matrix a, float c) {
  size_t row = blockIdx.y * blockDim.y + threadIdx.y;
  size_t col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= a.rows || col >= a.cols) {
    return;
  }
  MAT_AT(dst, row, col) = MAT_AT(a, row, col) * c;
}

void mat_scale_device(Matrix dst, const Matrix a, float c) {
  dim3 threads(32, 32);
  dim3 blocks((a.cols + threads.x - 1) / threads.x,
              (a.rows + threads.y - 1) / threads.y);
  hipLaunchKernelGGL(mat_scale_kernel, blocks, threads, 0, 0, dst, a, c);
}

__global__ void mat_lap_kernel(Matrix lap, const Matrix u, const float c) {
  size_t row = blockIdx.y * blockDim.y + threadIdx.y;
  size_t col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row == 0 || col == 0 || row == u.rows - 1 || col == u.cols - 1) {
    MAT_AT(lap, row, col) = 0.0f;
    return;
  }
  float sum = -4.0f * MAT_AT(u, row, col) + MAT_AT(u, row - 1, col) +
              MAT_AT(u, row, col - 1) + MAT_AT(u, row + 1, col) +
              MAT_AT(u, row, col + 1);
  MAT_AT(lap, row, col) = c * sum;
}

void mat_lap_device(Matrix lap, const Matrix u, const float c) {
  assert(u.rows == lap.rows);
  assert(u.cols == lap.cols);
  dim3 threads(32, 32);
  dim3 blocks((u.cols + threads.x - 1) / threads.x,
              (u.rows + threads.y - 1) / threads.y);
  hipLaunchKernelGGL(mat_lap_kernel, blocks, threads, 0, 0, lap, u, c);
}

__global__ void mat_tlap_kernel(Matrix next, const Matrix u, const float c) {
  size_t row = blockIdx.y * blockDim.y + threadIdx.y;
  size_t col = blockIdx.x * blockDim.x + threadIdx.x;

  size_t trow = threadIdx.y + 1;
  size_t tcol = threadIdx.x + 1;

  // tile + halo
  __shared__ float t[34][34];

  t[trow][tcol] = row < u.rows && col < u.cols ? MAT_AT(u, row, col) : 0.0f;

  if (trow == 1)
    t[0][tcol] = row > 0 && tcol < u.cols ? MAT_AT(u, row - 1, col) : 0.0f;
  if (trow == 32)
    t[33][tcol] =
        row <= u.rows && tcol < u.cols ? MAT_AT(u, row + 1, col) : 0.0f;
  if (tcol == 1)
    t[trow][0] = col > 0 && trow < u.rows ? MAT_AT(u, row, col - 1) : 0.0f;
  if (tcol == 32)
    t[trow][33] =
        col <= u.cols && trow < u.rows ? MAT_AT(u, row, col + 1) : 0.0f;

  __syncthreads();

  if (row >= u.rows || col >= u.cols)
    return;
  if (row == 0 || col == 0 || row == u.rows - 1 || col == u.cols - 1) {
    MAT_AT(next, row, col) = 0.0f;
    return;
  }
  float lap = -4.0f * t[trow][tcol] + t[trow - 1][tcol] + t[trow + 1][tcol] +
              t[trow][tcol + 1] + t[trow][tcol - 1];
  MAT_AT(next, row, col) = t[trow][tcol] + c * lap;
}

void mat_tlap_device(Matrix next, const Matrix u, const float c) {
  assert(u.rows == next.rows);
  assert(u.cols == next.cols);
  dim3 threads(32, 32);
  dim3 blocks((u.cols + threads.x - 1) / threads.x,
              (u.rows + threads.y - 1) / threads.y);
  hipLaunchKernelGGL(mat_tlap_kernel, blocks, threads, 0, 0, next, u, c);
}

__global__ void mat_wave_kernel(Matrix next, const Matrix u, const Matrix prev,
                                float c2) {
  size_t row = blockIdx.y * blockDim.y + threadIdx.y;
  size_t col = blockIdx.x * blockDim.x + threadIdx.x;

  size_t trow = threadIdx.y + 1;
  size_t tcol = threadIdx.x + 1;

  // tile + halo
  __shared__ float t[34][34];

  t[trow][tcol] = row < u.rows && col < u.cols ? MAT_AT(u, row, col) : 0.0f;

  if (trow == 1)
    t[0][tcol] = row > 0 && tcol < u.cols ? MAT_AT(u, row - 1, col) : 0.0f;
  if (trow == 32)
    t[33][tcol] =
        row <= u.rows && tcol < u.cols ? MAT_AT(u, row + 1, col) : 0.0f;
  if (tcol == 1)
    t[trow][0] = col > 0 && trow < u.rows ? MAT_AT(u, row, col - 1) : 0.0f;
  if (tcol == 32)
    t[trow][33] =
        col <= u.cols && trow < u.rows ? MAT_AT(u, row, col + 1) : 0.0f;

  __syncthreads();

  if (row >= u.rows || col >= u.cols)
    return;
  if (row == 0 || col == 0 || row == u.rows - 1 || col == u.cols - 1) {
    MAT_AT(next, row, col) = 0.0f;
    return;
  }
  float lap = -4.0f * t[trow][tcol] + t[trow - 1][tcol] + t[trow + 1][tcol] +
              t[trow][tcol + 1] + t[trow][tcol - 1];
  MAT_AT(next, row, col) =
      2 * t[trow][tcol] - MAT_AT(prev, row, col) + c2 * lap;
}

void mat_wave_device(Matrix next, const Matrix u, const Matrix prev, float c2) {
  assert(u.rows == next.rows);
  assert(u.cols == next.cols);
  dim3 threads(32, 32);
  dim3 blocks((u.cols + threads.x - 1) / threads.x,
              (u.rows + threads.y - 1) / threads.y);
  hipLaunchKernelGGL(mat_wave_kernel, blocks, threads, 0, 0, next, u, prev, c2);
}
