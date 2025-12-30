#include "cat.h"
#include <GL/gl.h>
#include <SDL2/SDL.h>
#include <SDL2/SDL_events.h>
#include <SDL2/SDL_mouse.h>
#include <SDL2/SDL_stdinc.h>
#include <SDL2/SDL_timer.h>
#include <SDL2/SDL_video.h>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <hip/amd_detail/amd_hip_vector_types.h>
#include <pthread.h>
#include <string>

#define COLOR
// #define READ_V

// physical constants
const float H_BAR = 0.05f;
const float Q = -0.1f;
const float M = 1.f;
const float B = 0.002f;

void init_window(SField H, SDL_Window *window, SDL_GLContext *glctx,
                 GLuint *tex);
void *write_frame(void *ARGS);
void cat_pulse(SField A, const float mx, const float my, const float2 k);

typedef struct {
  SField h;
  GLuint tex;
  SDL_Window *window;
} FrameData;

float2 barrier(float x, float y) {
  float r = ((x > 75 && x < 125) ? 0.5f : 0);
  return make_float2(r, 0);
}

float2 init_ax(float x, float y) { return make_float2(-0.5f * B * y, 0); }
float2 init_ay(float x, float y) {
  return make_float2(0, 0.5f * B * (x - 200));
}
float2 init_o(float x, float y) { return make_float2(0, 0); }

float2 playground(float x, float y) {
  return make_float2(x * x * y + y * y * y, 0);
}

SField init_scalar_potential(SField h) {
  SField O = cat_adapt(h);
  cat_host_fill(h, init_o);
  cat_to_device(O, h);
  return O;
}

VField init_vector_potential(SField h) {
  SField X = cat_adapt(h);
  cat_host_fill(h, init_ax);
  cat_to_device(X, h);
  SField Y = cat_adapt(h);
  cat_host_fill(h, init_ay);
  cat_to_device(Y, h);
  return {X, Y};
}

void on_click(SField h, SField T, SField U, int x0, int y0, int x, int y) {
  int dy = y - y0;
  int dx = x - x0;
  float n = sqrt(dy * dy + dx * dx);

  cat_to_host(h, U);
  // cat_pulse(h, x, y, 2.0f * make_float2(dx, dy) / n);
  // cat_pulse(h, 40, 25, make_float2(2.0f, 0));
  // cat_pulse(h, 150, 25, make_float2(-2.0f, 0));
  cat_pulse(h, 260, 25, make_float2(-2.0f, 0));
  cat_to_device(U, h);
  cat_to_device(T, h);
}

int main(int argc, char *argv[]) {
  const int cols = std::stoi(argv[1]);
  const int rows = std::stoi(argv[2]);
  const int MAX_STEP = std::stoi(argv[3]);
  const float DELTA_TIME = std::stof(argv[4]);

  // host matrix for reading / writing
  SField h = cat_alloc_host(rows, cols);

  // potiential matrices
  VField A = init_vector_potential(h);
  cat_scale(A.x, A.x, Q);
  cat_scale(A.y, A.y, Q);
  cat_preview(A.x);
  SField O = init_scalar_potential(h);
  cat_scale(O, O, Q);

  // buffer matrices
  SField U0 = cat_adapt(h);
  SField U1 = cat_adapt(h);
  SField U2 = cat_adapt(h);

  // initialization
  cat_fill(U0, 0);
  cat_fill(U1, 0);
  cat_fill(U2, 0);

  // user events
  bool running = true;
  SDL_Event e;
  int x0 = -1, y0 = -1, x1, y1, vx, vy;
  float n;

  // intitalize window
  SDL_Init(SDL_INIT_VIDEO);
  SDL_Window *window = SDL_CreateWindow("Wave Sim", SDL_WINDOWPOS_CENTERED,
                                        SDL_WINDOWPOS_CENTERED, h.cols, h.rows,
                                        SDL_WINDOW_OPENGL);
  SDL_GLContext glctx = (SDL_GL_CreateContext(window));
  SDL_GL_SetSwapInterval(1);

  GLuint tex;
  glGenTextures(1, &tex);
  glBindTexture(GL_TEXTURE_2D, tex);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, h.cols, h.rows, 0, GL_RGB,
               GL_UNSIGNED_BYTE, nullptr);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

  FrameData *frame_data = (FrameData *)malloc(sizeof(FrameData));
  frame_data->tex = tex;
  frame_data->window = window;

  SField A2 = cat_adapt(h);
  cfd_dot(A2, A, A);
  cat_scale(A2, A2, 0.5f / M);

  SField VA = cat_adapt(h);
  cfd_div(VA, A);
  cat_scale(VA, VA, make_float2(0, 0.5f * H_BAR / M));

  VField VU = {cat_adapt(h), cat_adapt(h)};
  SField AVU = cat_adapt(h);
  float2 AV_FACTOR = make_float2(0, -H_BAR / M);

  SField V2U = cat_adapt(h);
  float2 V2_FACTOR = make_float2(-0.5f * H_BAR * H_BAR / M, 0);

  SField H = cat_adapt(h);
  SField HU = cat_adapt(h);
  float2 H_FACTOR = make_float2(0, -2 * DELTA_TIME / H_BAR);
  float2 *tmp;

  while (running) {
    cat_to_host(h, U1);
    frame_data->h = h;

    write_frame(frame_data);

    for (int i = 0; i < MAX_STEP; i++) {
      // HU += [(ihV.(qA) + (qA)2)/2m + O]U
      cat_add(H, O, A2);
      cat_add(H, H, VA);
      cat_mult(HU, H, U1);

      // HU += -ih(qA).VU/m
      cat_grad(VU, U1);
      cfd_dot(AVU, A, VU);
      cat_scale(AVU, AVU, AV_FACTOR);
      cat_add(HU, HU, AVU);

      // HU += -h2V2U/2m
      cat_lap(V2U, U1);
      cat_scale(V2U, V2U, V2_FACTOR);
      cat_add(HU, HU, V2U);

      // U+ = U- - 2iHdt/h U
      cat_scale(HU, HU, H_FACTOR);
      cat_add(U2, U0, HU);
      cat_filter(U2);

      tmp = U0.data;
      U0.data = U1.data;
      U1.data = U2.data;
      U2.data = tmp;
    }

    while (SDL_PollEvent(&e)) {
      if (e.type == SDL_QUIT)
        running = false;
      if (e.type == SDL_MOUSEBUTTONDOWN)
        SDL_GetMouseState(&x0, &y0);
      if (e.type != SDL_MOUSEBUTTONUP)
        continue;
      SDL_GetMouseState(&x1, &y1);

      if (x0 == -1 || x0 == x1 || y0 == y1)
        continue;
      on_click(h, U0, U1, x0, y0, x1, y1);
    }

#ifdef READ_V
    SDL_GetMouseState(&vx, &vy);
    cat_to_host(H, V0);
    float v0 = CAT_AT(H, vy, vx).x;
    printf("V=%.2f x=%d y=%d", v0, vx, vy);
#endif // READ_V
  }

  // free matrices
  cat_free_host(h);
  cat_free_device(U0);
  cat_free_device(U1);
  cat_free_device(U2);
  cat_free_device(V2U);
  cat_free_device(VU.x);
  cat_free_device(VU.y);
  cat_free_device(A2);
  cat_free_device(VA);
  cat_free_device(A.x);
  cat_free_device(A.y);
  cat_free_device(O);
  cat_free_device(H);
  cat_free_device(HU);
  // free(frame_data);

  SDL_GL_DeleteContext(glctx);
  SDL_DestroyWindow(window);
  SDL_Quit();
  return 0;
}

void init_window(SField H, SDL_Window *window, SDL_GLContext *glctx,
                 GLuint *tex) {
  SDL_Init(SDL_INIT_VIDEO);
  window = SDL_CreateWindow("Wave Sim", SDL_WINDOWPOS_CENTERED,
                            SDL_WINDOWPOS_CENTERED, H.cols, H.rows,
                            SDL_WINDOW_OPENGL);
  *glctx = SDL_GL_CreateContext(window);
  SDL_GL_SetSwapInterval(1);

  glGenTextures(1, tex);
  glBindTexture(GL_TEXTURE_2D, *tex);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, H.cols, H.rows, 0, GL_RGB,
               GL_UNSIGNED_BYTE, nullptr);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
}

void cat_rgb(unsigned char *img, const SField u, const float scale) {
#pragma omp parallel for
  for (size_t row = 0; row < u.rows; row++) {
#pragma omp parallel for
    for (size_t col = 0; col < u.cols; col++) {
      float2 v = CAT_AT(u, row, col);
      const float hue = atan2f(v.y, v.x) / M_PI / 2 + 0.5f;
      // const float mag = fminf(sqrtf(v.x * v.x + v.y * v.y) * scale, 1);
      const float mag =
          fmax(0, fminf(sqrtf(v.x * v.x + v.y * v.y) * scale, 1) - 0.05f);

      const float h = hue * 6.0f;
      const float f = h - floorf(h);
      float3 rgb;
      switch ((int)h) {
      case 0:
        rgb = make_float3(1, f, 0);
        break;
      case 1:
        rgb = make_float3(1 - f, 1, 0);
        break;
      case 2:
        rgb = make_float3(0, 1, f);
        break;
      case 3:
        rgb = make_float3(0, 1 - f, 1);
        break;
      case 4:
        rgb = make_float3(f, 0, 1);
        break;
      case 5:
        rgb = make_float3(1, 0, 1 - f);
        break;
      }
      int idx = (row * u.cols + col) * 3;
#ifdef COLOR
      img[idx + 0] = (unsigned char)(255 * rgb.x * mag);
      img[idx + 1] = (unsigned char)(255 * rgb.y * mag);
      img[idx + 2] = (unsigned char)(255 * rgb.z * mag);
#else
      img[idx + 0] = (unsigned char)(255 * mag);
      img[idx + 1] = (unsigned char)(255 * mag);
      img[idx + 2] = (unsigned char)(255 * mag);
#endif // COLOR
    }
  }
}

void cat_pulse(SField A, const float mx, const float my, const float2 k) {
  for (int x = 0; x < A.cols; x++) {
    for (int y = 0; y < A.rows; y++) {
      float dx = x - mx;
      float dy = y - my;
      float sigma = 4.0f;
      float amp = 0.9f * expf(-(dx * dx + dy * dy) / (2 * sigma * sigma));
      float phi = k.x * dx + k.y * dy;
      CAT_AT(A, A.rows - y, x) += make_float2(amp * cosf(phi), amp * sinf(phi));
    }
  }
}

void *write_frame(void *ARGS) {
  FrameData *args = (FrameData *)ARGS;
  SField h = args->h;
  float maxval = 0.5f;

  unsigned char img[h.rows * h.cols * 3];
  cat_rgb(img, h, maxval);
  glBindTexture(GL_TEXTURE_2D, args->tex);
  glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, h.cols, h.rows, GL_RGB,
                  GL_UNSIGNED_BYTE, img);
  glClear(GL_COLOR_BUFFER_BIT);
  glEnable(GL_TEXTURE_2D);
  glBindTexture(GL_TEXTURE_2D, args->tex);
  glBegin(GL_QUADS);
  glTexCoord2f(0, 0);
  glVertex2f(-1, -1);
  glTexCoord2f(1, 0);
  glVertex2f(1, -1);
  glTexCoord2f(1, 1);
  glVertex2f(1, 1);
  glTexCoord2f(0, 1);
  glVertex2f(-1, 1);
  glEnd();
  SDL_GL_SwapWindow(args->window);

  // free(args);
  return NULL;
}
