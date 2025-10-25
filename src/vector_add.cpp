#include <hip/hip_runtime.h>
#include <iostream>

__global__ void vector_add(float *a, float *b, float *c, int N) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < N) {
    c[i] = a[i] + b[i];
  }
}

int main() {
  int N = 1 << 30;
  size_t size = N * sizeof(float);

  float *h_a = new float[N];
  float *h_b = new float[N];
  float *h_c = new float[N];

  for (int i = 0; i < N; i++) {
    h_a[i] = 1.0f;
    h_b[i] = 2.0f;
  }

  float *d_a, *d_b, *d_c;
  hipMalloc(&d_a, size);
  hipMalloc(&d_b, size);
  hipMalloc(&d_c, size);

  hipMemcpy(d_a, h_a, size, hipMemcpyHostToDevice);
  hipMemcpy(d_b, h_b, size, hipMemcpyHostToDevice);

  int threads = 256;
  int blocks = (N + threads - 1) / threads;
  hipLaunchKernelGGL(vector_add, dim3(blocks), dim3(threads), 0, 0, d_a, d_b,
                     d_c, N);

  hipMemcpy(h_c, d_c, size, hipMemcpyDeviceToHost);

  std::cout << "h_c[0] = " << h_c[0] << std::endl;

  hipFree(d_a);
  hipFree(d_b);
  hipFree(d_c);

  delete[] h_a;
  delete[] h_b;
  delete[] h_c;

  return 0;
}
