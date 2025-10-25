#include "mat.h"
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

Matrix mat_adapt_device(Matrix m) {
  Matrix d = mat_alloc_device(m.rows, m.cols);
  return d;
}

void write_frame(Matrix h, unsigned char *img, int step) {
  float maxval = 0;
  for (size_t i = 0; i < h.rows * h.cols; i++)
    if (maxval < h.data[i])
      maxval = h.data[i];
  if (maxval == 0)
    maxval = 1;

  for (int row = 0; row < h.rows; row++) {
    for (int col = 0; col < h.cols; col++) {
      float v = MAT_AT(h, row, col) / maxval;
      img[row * h.cols + col] =
          (unsigned char)(255.0f * fmin(fmax(v, 0.0f), 1.0f));
    }
  }

  char filename[64];
  sprintf(filename, "pngs/frame_%08d.png", step);
  stbi_write_png(filename, h.cols, h.rows, 1, img, h.cols);
  printf("Saved %08d\n", step);
}

int main(int argc, char *argv[]) {
  hipEvent_t start, stop;
  (void)hipEventCreate(&start);
  (void)hipEventCreate(&stop);
  float ms = 0.0f;

  size_t dim = 1000;
  int cycle = 100000;
  float diffusion = 1.0;
  float delta_time = 0.1f;

  Matrix A = mat_alloc_host(dim, dim);
  Matrix dA = mat_adapt_device(A);
  Matrix dB = mat_adapt_device(A);
  Matrix dC = mat_adapt_device(A);

  unsigned char *img = (unsigned char *)malloc(A.rows * A.cols);
  // initialization
  mat_fill_host(A, 0);
  MAT_AT(A, dim / 2, dim / 2) = 10;
  mat_preview(A);
  mat_copy_to_device(dA, A);
  mat_clone_device(dB, dA);

  (void)hipEventRecord(start, 0);
  for (int i = 0; i < cycle; i++) {
    mat_wave_device(dC, dB, dA, 0.001f);
    float *tmp = dA.data;
    dA.data = dB.data;
    dB.data = dC.data;
    dC.data = tmp;
    if (i % 50 != 0)
      continue;
    mat_copy_to_host(A, dA);
    write_frame(A, img, i);
  }
  (void)hipEventRecord(stop, 0);
  (void)hipEventSynchronize(stop);
  (void)hipEventElapsedTime(&ms, start, stop);
  printf("Lap %d cycles: %.1f ms!!!\n", cycle, ms);
  mat_copy_to_host(A, dA);
  mat_preview(A);

  free(img);
  mat_free_host(A);
  mat_free_device(dA);
  mat_free_device(dC);
  return 0;
}
