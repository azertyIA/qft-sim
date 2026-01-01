#include "cat.h"
#include "quantum.h"
#include <GL/gl.h>
#include <SDL2/SDL.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

// #define COLOR
// #define PNCOLOR

// physical constants
const float H_BAR = 0.05f;
const float Q = -0.1f;
const float M = 1.f;
const float B = -0.06f;
const float E = -0.0000005f;

const int FRAME_ROWS = 2;
const int FRAME_COLS = 3;

typedef struct {
  SField h[FRAME_ROWS * FRAME_COLS];
  float s[FRAME_ROWS * FRAME_COLS];
  GLuint tex;
  SDL_Window *window;
} FrameData;

void init_window(SField H, SDL_Window *window, SDL_GLContext *glctx,
                 GLuint *tex);
void write_frame(FrameData *args, unsigned char *img);
void cat_pulse(SField A, const float mx, const float my, const float2 k);

float2 init_o(float x, float y) {
  // float r = x < 60 ? E : x < 68 ? E / 2 : 0;
  float r = 0.5f * E * (128 - x);
  // float r = 0;
  // float i = 0.0001f;
  float i = 0;
  return make_float2(r, i);
}

float2 init_ax(float x, float y) {
  // float r = 0;
  float r = -0.5f * B * (y - 64);
  float i = 0;
  return make_float2(r, i);
}

float2 init_ay(float x, float y) {
  // float r = 0;
  float r = 0.5f * B * (x - 64);
  float i = 0;
  return make_float2(r, i);
}

void init_window(SField H, SDL_Window *&window, SDL_GLContext &glctx,
                 GLuint &tex);

void on_click(SField h, int x0, int y0, int x, int y) {
  int dy = y - y0;
  int dx = x - x0;
  float n = sqrt(dy * dy + dx * dx);
  cat_pulse(h, x, y, 2.0f * make_float2(dx, dy) / n);
  cat_pulse(h, 128 - x, 128 - y, -2.0f * make_float2(dx, dy) / n);
}

int main(int argc, char *argv[]) {
  const int cols = std::stoi(argv[1]);
  const int rows = std::stoi(argv[2]);
  const int MAX_STEP = std::stoi(argv[3]);
  const float DELTA_TIME = std::stof(argv[4]);

  // host matrix for reading / writing
  SField h = s_alloc_host(rows, cols);
  SField h0 = s_alloc_host(rows, cols);
  SField h1 = s_alloc_host(rows, cols);
  SField h2 = s_alloc_host(rows, cols);
  SField h3 = s_alloc_host(rows, cols);
  SField h4 = s_alloc_host(rows, cols);

  // user events
  bool running = true;
  SDL_Event e;
  int x0 = -1, y0 = -1, x1, y1, vx, vy;

  // intitalize window
  SDL_Window *window;
  SDL_GLContext glctx;
  GLuint tex;
  init_window(h, window, glctx, tex);

  FrameData *frame_data = (FrameData *)malloc(sizeof(FrameData));
  frame_data->tex = tex;
  frame_data->window = window;
  frame_data->h[3] = h;
  frame_data->s[3] = 1.0f;

  frame_data->h[0] = h3;
  frame_data->s[0] = 30.f / H_BAR / sqrtf(M);

  frame_data->h[5] = h4;
  frame_data->s[5] = 0.5f / abs(Q * B);

  frame_data->h[4] = h2;
  frame_data->s[4] = 1.0f / sqrtf(abs(Q * (B + E)));

  frame_data->h[2] = h0;
  frame_data->s[2] = 300.f;

  frame_data->h[1] = h1;
  frame_data->s[1] = 300.f;

  unsigned char img[h.rows * h.cols * 3 * FRAME_ROWS * FRAME_COLS];

  QuantumGaugeParams qgp = {H_BAR, M, Q, DELTA_TIME, init_ax, init_ay, init_o};
  QuantumGauge gauge = qg_alloc(h);
  qg_init(gauge, qgp, h);

  while (running) {
    qg_dump(h, gauge);
    s_to_host(h0, gauge.AVU);
    s_to_host(h1, gauge.VA);
    s_to_host(h2, gauge.H);
    s_to_host(h3, gauge.V2U);
    s_to_host(h4, gauge.A.x);

    write_frame(frame_data, img);
    for (int i = 0; i < MAX_STEP; i++)
      qg_step_decomp(gauge);

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
      qg_dump(h, gauge);
      on_click(h, x0, y0, x1, y1);
      qg_load(gauge, h);
    }
  }

  // free matrices
  s_free_host(h);
  qg_free(gauge);

  SDL_GL_DeleteContext(glctx);
  SDL_DestroyWindow(window);
  SDL_Quit();
  return 0;
}

void init_window(SField H, SDL_Window *&window, SDL_GLContext &glctx,
                 GLuint &tex) {
  SDL_Init(SDL_INIT_VIDEO);
  window = SDL_CreateWindow("Wave Sim", SDL_WINDOWPOS_CENTERED,
                            SDL_WINDOWPOS_CENTERED, H.cols * FRAME_COLS,
                            H.rows * FRAME_ROWS, SDL_WINDOW_OPENGL);
  glctx = (SDL_GL_CreateContext(window));
  SDL_GL_SetSwapInterval(1);

  glGenTextures(1, &tex);
  glBindTexture(GL_TEXTURE_2D, tex);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, H.cols * FRAME_COLS,
               H.rows * FRAME_ROWS, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
}

void cat_rgb(unsigned char *img, const SField u, const float scale,
             const int frow, const int fcol) {
  int fdx = frow * (u.cols * FRAME_COLS * u.rows) + fcol * u.cols;
#pragma omp parallel for
  for (size_t row = 0; row < u.rows; row++) {
#pragma omp parallel for
    for (size_t col = 0; col < u.cols; col++) {
      int idx = (fdx + row * (u.cols * FRAME_COLS) + col) * 3;
      // int idx = ((u.cols * FRAME_COLS * FRAME_ROWS) +
      //            row * (u.cols * FRAME_COLS) + col) *
      //           3;
      float2 v = CAT_AT(u, row, col) * scale;
      float mag = fminf(sqrtf(v.x * v.x + v.y * v.y), 1);
#if defined COLOR
      const float hue = (atan2f(v.y, v.x) + 0.000001f) / M_PI / 2 + 0.5f;

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
      case 6:
        rgb = make_float3(1, f, 0);
        break;
      }
      img[idx + 0] = (unsigned char)(255 * rgb.x * mag);
      img[idx + 1] = (unsigned char)(255 * rgb.y * mag);
      img[idx + 2] = (unsigned char)(255 * rgb.z * mag);
#elif defined PNCOLOR
      float3 rgb = v.x > 0 ? make_float3(1, 1, 0) : make_float3(1, 0, 1);
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

void write_frame(FrameData *args, unsigned char *img) {
  SField h = args->h[0];

#pragma omp parallel for
  for (int frow = 0; frow < FRAME_ROWS; frow++) {
#pragma omp parallel for
    for (int fcol = 0; fcol < FRAME_COLS; fcol++) {
      int fidx = FRAME_COLS * frow + fcol;
      // int fidx = 0;
      cat_rgb(img, args->h[fidx], args->s[fidx], frow, fcol);
    }
  }
  glBindTexture(GL_TEXTURE_2D, args->tex);
  glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, h.cols * FRAME_COLS,
                  h.rows * FRAME_ROWS, GL_RGB, GL_UNSIGNED_BYTE, img);
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
}

void cat_pulse(SField A, const float mx, const float my, const float2 k) {
  float dy, dx, sigma = 4.0f, amp, phi;
  for (int x = 0; x < A.cols; x++) {
    for (int y = 0; y < A.rows; y++) {
      dx = x - mx;
      dy = y - my;
      amp = 0.9f * expf(-(dx * dx + dy * dy) / (2 * sigma * sigma));
      phi = k.x * dx + k.y * dy;
      CAT_AT(A, A.rows - y, x) += make_float2(amp * cosf(phi), amp * sinf(phi));
    }
  }
}
