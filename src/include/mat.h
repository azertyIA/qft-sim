#pragma once
#include <hip/hip_runtime.h>
#include <stddef.h>

typedef struct {
  size_t rows, cols, stride;
  float *data;
} Matrix;

#define MAT_AT(m, i, j) (m).data[(i) * (m).stride + (j)]

Matrix mat_alloc_host(size_t rows, size_t cols);
Matrix mat_alloc_device(size_t rows, size_t cols);
Matrix mat_adapt(const Matrix d);

void mat_free_host(Matrix m);
void mat_free_device(Matrix m);

void mat_copy_to_device(Matrix d, Matrix h);
void mat_copy_to_host(Matrix h, Matrix d);

void mat_clone_device(Matrix dst, const Matrix a);
void mat_preview(Matrix m);

void mat_fill_host(Matrix m, float value);
void mat_fill_device(Matrix m, float value);

void mat_scale_device(Matrix dst, const Matrix a, float c);

void mat_add_host(Matrix dst, const Matrix a, const Matrix b);
void mat_add_device(Matrix dst, const Matrix a, const Matrix b);

void mat_mult_host(Matrix dst, const Matrix a, const Matrix b);
void mat_mult_device(Matrix dst, const Matrix a, const Matrix b);

void mat_lap_device(Matrix dst, const Matrix u, const float c);
void mat_tlap_device(Matrix next, const Matrix u, const float c);
void mat_wave_device(Matrix next, const Matrix u, const Matrix prev, float c2);
